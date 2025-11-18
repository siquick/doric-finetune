#!/usr/bin/env python3
"""
Scrape Doric phrases dictionary from doricphrases.com
Extracts Doric-English word/phrase pairs for all letters A-Z
"""

import json
import string
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


BASE_URL = "https://doricphrases.com/phrases.php"
RESULTS_PER_PAGE = 20


def extract_phrases_from_page(html_content: str) -> list[dict[str, str]]:
    """
    Extract Doric-English phrase pairs from HTML content.

    Structure: Doric phrase is in <h1> tag, English translation is in next <p> sibling.

    Returns list of dicts with 'doric' and 'english' keys.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    phrases = []

    # Find all h1 tags which contain the Doric phrases
    h1_tags = soup.find_all("h1")

    for h1 in h1_tags:
        doric = h1.get_text(strip=True)

        # Skip if it's empty or looks like a header/navigation element
        if not doric or len(doric) > 200:  # Likely not a phrase if too long
            continue

        # Find the next sibling which should be a <p> tag with the English translation
        next_sibling = h1.find_next_sibling()

        if next_sibling and next_sibling.name == "p":
            english = next_sibling.get_text(strip=True)

            if english:  # Ensure it's not empty
                phrases.append({"doric": doric, "english": english})

    return phrases


def has_next_page(html_content: str) -> bool:
    """Check if there's a 'Next' button/link on the page."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Look for "Next" link - it's an <a> tag with text "Next"
    next_link = soup.find("a", string=lambda text: text and text.strip() == "Next")

    if next_link:
        href = next_link.get("href", "")
        # Check if it's a valid link (not javascript:void or empty)
        if href and "javascript:void" not in href.lower() and href.strip():
            return True

    return False


def scrape_letter(client: httpx.Client, letter: str) -> list[dict[str, str]]:
    """
    Scrape all pages for a given letter.

    Returns list of all phrase pairs for that letter.
    """
    all_phrases = []
    page = 1

    print(f"Scraping letter '{letter}'...")

    while True:
        url = f"{BASE_URL}?p={page}&s={RESULTS_PER_PAGE}&l={letter}"
        print(f"  Fetching page {page}...", end=" ")

        try:
            response = client.get(url, timeout=30.0)
            response.raise_for_status()

            phrases = extract_phrases_from_page(response.text)

            if phrases:
                all_phrases.extend(phrases)
                print(f"Found {len(phrases)} phrases (total: {len(all_phrases)})")
            else:
                print("No phrases found")

            # Check if there's a next page
            if not has_next_page(response.text):
                print(f"  No more pages for letter '{letter}'")
                break

            page += 1

        except httpx.HTTPError as e:
            print(f"Error fetching page {page}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error on page {page}: {e}")
            break

    return all_phrases


def main():
    """Main scraping function."""
    output_file = Path(__file__).parent.parent / "datasets" / "doric_dictionary.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_phrases = []

    # Create HTTP client with proper headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        # Iterate through all letters A-Z
        for letter in string.ascii_uppercase:
            phrases = scrape_letter(client, letter)
            all_phrases.extend(phrases)
            print(f"Completed letter '{letter}': {len(phrases)} phrases\n")

    # Save results
    print(f"\nTotal phrases collected: {len(all_phrases)}")
    print(f"Saving to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_phrases, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()
