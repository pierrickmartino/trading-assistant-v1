import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Sépare d'abord le Markdown en endpoints (chaque section commence par '##' 
    sur une ligne seule). Puis, si un endpoint dépasse 'chunk_size', on le découpe 
    en plus petits blocs en respectant les blocs de code, les paragraphes et les phrases.
    """
    # --- 1) Séparation en endpoints par '##' ---
    lines = text.splitlines()
    endpoints = []
    current_section = []
    found_endpoint = False
    
    for line in lines:
        # On repère le début d'un nouveau endpoint si la ligne est strictement '##'
        if line.strip() == "##":
            # On "clôt" la section précédente s'il y en avait une
            if current_section:
                endpoints.append("\n".join(current_section).strip())
                current_section = []
            current_section.append(line)
            found_endpoint = True
        else:
            # On n'accumule que si on a trouvé au moins un '##'
            # (autrement, tout le texte avant le 1er '##' est ignoré)
            if found_endpoint:
                current_section.append(line)
    
    # Ne pas oublier la dernière section
    if current_section:
        endpoints.append("\n".join(current_section).strip())
    
    # --- 2) Découpage de chaque endpoint en chunks plus petits si nécessaire ---
    final_chunks = []
    for endpoint in endpoints:
        if len(endpoint) <= chunk_size:
            # Si le endpoint fait moins que chunk_size, on l'ajoute directement
            final_chunks.append(endpoint)
        else:
            # Sinon, on le découpe via la logique code/paragraphes/phrases
            final_chunks.extend(_split_large_text(endpoint, chunk_size))
    
    return final_chunks

def _split_large_text(text: str, chunk_size: int) -> List[str]:
    """
    Découpe un texte trop grand en chunks en suivant un ordre de priorité :
    - couper juste avant un bloc de code (```),
    - sinon couper avant un paragraphe vide (\n\n),
    - sinon couper à la fin d’une phrase ('. '),
    - sinon on coupe "comme on peut" si on n'a rien trouvé.
    """
    chunks = []
    start = 0
    length = len(text)
    
    while start < length:
        end = start + chunk_size
        if end >= length:
            # Dernier bout du texte
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        
        chunk_candidate = text[start:end]
        
        # 1) Tenter de rompre juste avant le dernier ``` si on le trouve > 30% du chunk
        code_boundary = chunk_candidate.rfind('```')
        if code_boundary != -1 and code_boundary > chunk_size * 0.3:
            end = start + code_boundary
        
        # 2) Sinon, essayer de trouver un saut de paragraphe \n\n
        elif '\n\n' in chunk_candidate:
            last_break = chunk_candidate.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        
        # 3) Sinon, couper au dernier point ('. ')
        elif '. ' in chunk_candidate:
            last_period = chunk_candidate.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1
        
        # On prend le texte jusqu'à 'end'
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # On se décale : on démarre le chunk suivant juste après la fin du chunk précédent
        # (ou +1 pour éviter de bloquer si end = start)
        start = max(start + 1, end)
    
    return chunks

# def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
#     """Split text into chunks, respecting code blocks and paragraphs."""
#     chunks = []
#     start = 0
#     text_length = len(text)

#     while start < text_length:
#         # Calculate end position
#         end = start + chunk_size

#         # If we're at the end of the text, just take what's left
#         if end >= text_length:
#             chunks.append(text[start:].strip())
#             break
        
#         # Try to find the start of an endpoint section (##)
#         chunk = text[start:end]
#         code_block = chunk.rfind('##')
#         if code_block != -1 and code_block > chunk_size * 0.3:
#             end = start + code_block
        
#         # If no endpoint section, try to find a code block boundary first (```)
#         elif '```' in chunk:
#             last_boundary = chunk.rfind('```')
#             if last_boundary > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
#                 end = start + last_boundary

#         # If no code block, try to break at a paragraph
#         elif '\n\n' in chunk:
#             # Find the last paragraph break
#             last_break = chunk.rfind('\n\n')
#             if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
#                 end = start + last_break

#         # If no paragraph break, try to break at a sentence
#         elif '. ' in chunk:
#             # Find the last sentence break
#             last_period = chunk.rfind('. ')
#             if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
#                 end = start + last_period + 1

#         # Extract chunk and clean it up
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)

#         # Move start position for next chunk
#         start = max(start + 1, end)

#     return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def is_pro_endpoint(chunk: str) -> str:
    """
    Détecte si un endpoint est un 'PRO' en cherchant la présence d'une image ![](...)
    juste après le titre (c.-à-d. soit sur la même ligne que 'Get ...', 
    soit la ligne immédiatement suivante).
    """
    # On découpe en lignes (on retire les lignes vides pour simplifier)
    lines = [l.strip() for l in chunk.splitlines() if l.strip()]
    
    # Pour chaque ligne, si elle commence par "Get " (ou "Get" insensible à la casse),
    # on regarde si elle contient déjà "![](...)". Sinon, on check la ligne suivante.
    for idx, line in enumerate(lines):
        if line.lower().startswith("get "):  # repère le titre
            # Cas 1 : l'image est sur la même ligne
            if "![](" in line:
                return 'PRO'
            # Cas 2 : l'image est à la ligne suivante
            if idx + 1 < len(lines):
                if "![](" in lines[idx + 1]:
                    return 'PRO'
            # Si ni sur la même ligne, ni la suivante, on considère que c'est un endpoint "gratuit"
            return 'FREE'
    
    # Si on n'a pas trouvé de ligne commençant par "Get ",
    # on ne sait pas détecter (on renvoie False par défaut)
    return 'N/A'

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)

    # Get PRO information
    pro_information = await is_pro_endpoint(chunk) 
    
    # Create metadata
    metadata = {
        "source": "polygonscan_api_docs",
        "chunk_size": len(chunk),
        "is_pro": pro_information,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_polygonscan_api_docs_urls() -> List[str]:
    """Get URLs from PolygonScan API docs sitemap."""
    sitemap_url = "https://docs.polygonscan.com/sitemap-pages.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Get URLs from PolygonScan API docs
    urls = get_polygonscan_api_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
