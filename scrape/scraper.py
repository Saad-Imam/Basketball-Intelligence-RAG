# Scraper for hoopstudent website 
import os
import re
import time
import requests
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

# 

def sanitize_filename(term_name):
    """
    Each term on the page will have a separate text file, with their term name as the file name
    below we sanitize the filename to ensure that the scraper doesn't face any errors due to punctuation marks
    """
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s\-]+', '_', term_name.strip())
    # Remove any character that isn't alphanumeric or an underscore
    name = re.sub(r'[^\w_]', '', name)
    return name

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
    Visits the detail page and extracts text strictly from the main article body.
    Captures paragraphs, lists, and image alt texts.
    """
    soup = get_soup(detail_url)
    if not soup:
        return "Failed to retrieve detailed explanation."
        
    # Strictly target the main article body
    article_body = soup.find('article') or soup.find(class_=re.compile(r'entry-content|post-content'))
    
    if not article_body:
        return "Could not locate main article content."

    content_parts =[]
    
    # Extract specific tags in order to keep the RAG context clean and sequential
    for element in article_body.find_all(['p', 'ul', 'ol', 'img', 'h2', 'h3']):
        
        # Extract Image Alt Text
        if element.name == 'img':
            alt_text = element.get('alt', '').strip()
            if alt_text:
                content_parts.append(f"[Image Alt Text: {alt_text}]")
                
        # Format Lists cleanly
        elif element.name in ['ul', 'ol']:
            for li in element.find_all('li'):
                content_parts.append(f"- {li.get_text(strip=True)}")
                
        # Format Text (Paragraphs, Headings)
        else:
            text = element.get_text(strip=True)
            if text:
                content_parts.append(text)
                
    # Join everything into a cohesive detailed explanation
    return "\n\n".join(content_parts)

# --- MAIN ORCHESTRATION ---

def main():
    print("Starting HoopStudent Scraper for RAG...\n")
    
    for category_name, hub_url in CATEGORIES.items():
        print(f"=== Processing Category: {category_name} ===")
        
        # Create the category folder if it doesn't exist
        os.makedirs(category_name, exist_ok=True)
        
        # Step 1: Scrape Hub Page
        print(f"Fetching terms from {hub_url}...")
        terms = parse_hub_page(hub_url)
        print(f"Found {len(terms)} terms in {category_name}.\n")
        
        # Step 2: Iterate through terms, scrape details, and save
        for idx, term_data in enumerate(terms, 1):
            term_name = term_data['term']
            concise_def = term_data['concise_def']
            detail_url = term_data['url']
            
            filename = sanitize_filename(term_name) + ".txt"
            filepath = os.path.join(category_name, filename)
            
            print(f"[{idx}/{len(terms)}] Scraping detail for: {term_name}")
            
            # Scrape detailed page
            detailed_explanation = extract_detailed_content(detail_url)
            
            # Step 3: Format and save the data
            file_content = f"Term: {term_name}\n\n"
            file_content += f"--- Concise Definition ---\n{concise_def}\n\n"
            file_content += f"--- Detailed Explanation ---\n{detailed_explanation}\n"
            
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(file_content)
                
            # Politeness delay: wait 1.5 seconds between page loads to avoid being blocked
            time.sleep(1.5)
            
        print(f"Finished processing {category_name}!\n")

    print("Scraping completed successfully! Your dataset is ready.")

if __name__ == "__main__":
    main()