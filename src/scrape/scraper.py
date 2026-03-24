# Scraper for hoopstudent website 
import os
import re
import time
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://hoopstudent.com"
CATEGORIES = {
    "Glossary": "https://hoopstudent.com/basketball-glossary/",
    "Offense": "https://hoopstudent.com/offense/",
    "Defense": "https://hoopstudent.com/defense/"
}

# standard User-Agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

 # Updated Output Path
OUTPUT_DIR = "corpus/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "layer2_rulebook_chunks.json")

def sanitize_filename(term_name):
    """
    Each term on the page will have a separate text file, with their term name as the file name
    below we sanitize the filename to ensure that the scraper doesn't face any errors due to punctuation marks
    """
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s\-]+', '_', term_name.strip())
    # Remove any character that isn't alphanumeric or an underscore
    name = re.sub(r'[^\w_]', '', name)
    return name.lower()

def get_soup(url):
    """
    Fetch a URL and return a BeautifulSoup object
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"  [!] Error fetching {url}: {e}")
        return None

def parse_hub_page(hub_url):
    """
    Parses main hub page (Glossary, Offense, Defense) to extract the term, concise definition, and detailed URL.
    Returns a list of dictionaries.
    """
    soup = get_soup(hub_url)
    if not soup:
        return []

    terms_data =[]
    
    # Target the main content area of the hub page to avoid sidebar/footer links
    # WordPress sites typically use 'entry-content' or 'post-content'
    main_content = soup.find(class_=re.compile(r'entry-content|post-content'))
    if not main_content:
        main_content = soup # Fallback to the whole page if specific container isn't found
        
    # We look for blocks holding definitions (paragraphs, list items, headings)
    blocks = main_content.find_all(['p', 'li', 'h3', 'h4'])
    
    for block in blocks:
        link = block.find('a', href=True)
        if link:
            term = link.get_text(strip=True)
            url = urljoin(BASE_URL, link['href'])
            
            # Filter out non-term links (like "Read More", internal category tags, or empty text)
            if not term or "hoopstudent.com" not in url or "category" in url or "tag" in url:
                continue
                
            # Extract the concise definition
            concise_def = block.get_text(" ", strip=True)
            
            # Heuristic: If the block was just a heading, the definition is likely in the next paragraph
            if block.name in ['h3', 'h4'] or len(concise_def) < len(term) + 10:
                next_p = block.find_next_sibling('p')
                if next_p:
                    concise_def += " - " + next_p.get_text(" ", strip=True)
            
            # Avoid duplicate URLs on the same hub page
            if url not in [item['url'] for item in terms_data]:
                terms_data.append({
                    "term": term,
                    "concise_def": concise_def,
                    "url": url
                })
                
    return terms_data

def extract_detailed_content(detail_url):
    """
    Extracts content dynamically into sections based on H2/H3 headers.
    Returns the combined text AND the structured section list.
    """
    soup = get_soup(detail_url)
    if not soup:
        return "",[]
        
    article_body = soup.find('article') or soup.find(class_=re.compile(r'entry-content|post-content'))
    
    if not article_body:
        return "",[]

    sections = []
    current_heading = "Introduction"
    current_parts =[]

    for element in article_body.find_all(['p', 'ul', 'ol', 'img', 'h2', 'h3']):
        # When hitting a new heading, save the previous section
        if element.name in ['h2', 'h3']:
            if current_parts:
                section_text = "\n\n".join(current_parts).strip()
                # Skip Table of Contents sections
                if not section_text.startswith("Table of Contents"):
                    sections.append({
                        "heading": current_heading,
                        "text": section_text
                    })
            current_heading = element.get_text(strip=True)
            current_parts = []
            
        # Extract Image Alt Text
        elif element.name == 'img':
            alt_text = element.get('alt', '').strip()
            if alt_text:
                current_parts.append(f"[Image Alt Text: {alt_text}]")
                
        # Format Lists cleanly
        elif element.name in ['ul', 'ol']:
            for li in element.find_all('li'):
                current_parts.append(f"- {li.get_text(strip=True)}")
                
        # Format Text (Paragraphs)
        else:
            text = element.get_text(strip=True)
            if text:
                current_parts.append(text)

    # Don't forget to append the final section
    if current_parts:
        sections.append({
            "heading": current_heading,
            "text": "\n\n".join(current_parts).strip()
        })

    # Combine all section text for the 'detailed_explanation' field
    full_text = "\n\n".join([s['text'] for s in sections])
    return full_text, sections

def main():
    print("Starting HoopStudent JSON Scraper for Layer 2...\n")
    
    # Initialize a single list to hold all scraped terms
    all_layer2_data = []

    for category_name, hub_url in CATEGORIES.items():
        print(f"=== Processing Category: {category_name} ===")
        
        print(f"Fetching terms from {hub_url}...")
        terms = parse_hub_page(hub_url)
        print(f"Found {len(terms)} terms in {category_name}.\n")
        
        for idx, term_data in enumerate(terms, 1):
            term_name = term_data['term']
            concise_def = term_data['concise_def']
            detail_url = term_data['url']
            
            safe_name = sanitize_filename(term_name)
            print(f"[{idx}/{len(terms)}] Scraping detail for: {term_name}")
            
            # Scrape and chunk detailed page
            detailed_explanation, sections = extract_detailed_content(detail_url)
            
            # Structure the JSON document
            doc = {
                "doc_id": f"hoopstudent_{category_name.lower()}_{safe_name}",
                "term": term_name,
                "category": category_name,
                "source_url": detail_url,
                "layer": 2, 
                "source_site": "hoopstudent",
                "concise_definition": concise_def,
                "sections": sections
            }
            
            # Append to our master list instead of saving individual files
            all_layer2_data.append(doc)
                
            # Politeness delay
            time.sleep(1.5)
            
        print(f"Finished gathering {category_name}!\n")

    # --- FINAL SAVE STEP ---
    # Ensure the directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Write the entire list to one single JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_layer2_data, f, ensure_ascii=False, indent=2)

    print(f"Scraping completed! Saved {len(all_layer2_data)} terms to: {OUTPUT_FILE}")
if __name__ == "__main__":
    main()