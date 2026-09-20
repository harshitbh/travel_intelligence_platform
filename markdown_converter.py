#!/usr/bin/env python3
"""
Deep context scraper for a Singapore travel guide.
It follows internal links and extracts complete content from travel pages.
Run: python markdown_converter.py
"""

# Standard library imports
import os
import time
from urllib.parse import urljoin, urlparse

# Third-party imports
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from playwright.sync_api import sync_playwright


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------
class DeepScraperSingapore:
    """Scrapes a main page and linked attraction or information pages."""

    def __init__(self):
        # Track pages already processed so the same content is not scraped twice
        self.visited_urls = set()

        # Base domains used for content collection
        self.base_urls = [
            "https://www.visitsingapore.com",
            "https://en.wikivoyage.org",
        ]

    # ---------------------------------------------------------------------
    # URL and content helpers
    # ---------------------------------------------------------------------
    def is_internal_link(self, url, base_domain):
        """Check if the URL belongs to the current site domain."""
        try:
            parsed = urlparse(url)
            return base_domain in parsed.netloc
        except Exception:
            return False

    def clean_html(self, html_content):
        """Remove navigation, footer, and other noise elements from page HTML."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove common page noise elements
        noise_tags = ["nav", "footer", "script", "style", "button"]
        for tag in noise_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove noisy classes commonly used for navigation or ads
        noise_classes = [
            "sidebar",
            "navigation",
            "header",
            "breadcrumb",
            "cookie-banner",
            "popup",
            "ad",
            "advertisement",
        ]
        for noise_class in noise_classes:
            for element in soup.find_all(class_=noise_class):
                element.decompose()

        # Prioritize the main content area when available
        main = soup.find("main")
        if not main:
            main = soup.find("article")
        if not main:
            main = soup.find(class_="mw-parser-output")
        if not main:
            main = soup.body if soup.body else soup

        return str(main) if main else html_content

    def extract_links(self, html_content, base_url):
        """Extract internal attraction and information links from a page."""
        soup = BeautifulSoup(html_content, "html.parser")
        links = []
        base_domain = urlparse(base_url).netloc

        skip_url_patterns = [
            "/de_de/", "/id_id/", "/ja_jp/", "/ko_kr/", "/vi_vn/", "/th_th/",
            "/cn/", "/zh/", "/search", "/login", "/privacy", "/terms",
        ]

        for link in soup.find_all("a", href=True):
            try:
                url = link.get("href", "")
                text = link.get_text(strip=True)

                if not url or not text or len(text) < 3:
                    continue

                if url.startswith("/"):
                    url = urljoin(base_url, url)
                elif url.startswith("#"):
                    continue

                if not self.is_internal_link(url, base_domain):
                    continue

                if any(pattern in url.lower() for pattern in skip_url_patterns):
                    continue

                skip_keywords = [
                    "home",
                    "menu",
                    "search",
                    "login",
                    "contact",
                    "privacy",
                    "terms",
                    "cookie",
                    "sitemap",
                    "facebook",
                    "instagram",
                    "twitter",
                    "youtube",
                    "apply",
                    "join",
                ]
                if any(keyword in text.lower() for keyword in skip_keywords):
                    continue

                valid_keywords = [
                    "attraction",
                    "thing-to-do",
                    "neighbourhood",
                    "restaurant",
                    "hotel",
                    "museum",
                    "temple",
                    "garden",
                    "activity",
                    "experience",
                    "tour",
                    "itinerary",
                    "culture",
                    "heritage",
                    "park",
                    "zoo",
                    "aquarium",
                    "shopping",
                    "dining",
                    "wellness",
                    "family",
                    "marina-bay",
                    "sentosa",
                    "kampong",
                    "orchard",
                    "chinatown",
                    "little-india",
                ]

                if any(keyword in url.lower() for keyword in valid_keywords):
                    if url not in self.visited_urls:
                        links.append((url, text))
            except Exception:
                continue

        return links

    # ---------------------------------------------------------------------
    # Page scraping
    # ---------------------------------------------------------------------
    def scrape_page(self, page, url, title=None, timeout=30000):
        """Scrape a single page and return markdown content."""
        try:
            page.goto(url, timeout=timeout, wait_until="networkidle")
            page.wait_for_timeout(500)

            html = page.content()
            cleaned_html = self.clean_html(html)
            markdown_content = md(cleaned_html)

            page_title = page.title() if not title else title

            return {
                "title": page_title,
                "url": url,
                "content": markdown_content,
                "status": "success",
            }
        except Exception as e:
            print(f"      Error scraping: {str(e)[:50]}")
            return None

    def scrape_deep(self, start_url, max_pages=50, max_depth=2):
        """Deep scrape from a start page through related internal links."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page_obj = browser.new_page()

            all_content = []
            queue = [(start_url, "Start Page", 0)]
            page_count = 0

            while queue and page_count < max_pages:
                current_url, current_title, depth = queue.pop(0)

                if current_url in self.visited_urls:
                    continue

                self.visited_urls.add(current_url)
                page_count += 1

                print(f"  [{page_count:2d}] {current_title}")

                result = self.scrape_page(page_obj, current_url, current_title)

                if result:
                    all_content.append(result)

                    if depth < max_depth:
                        try:
                            page_obj.goto(current_url, wait_until="networkidle")
                            html = page_obj.content()
                            links = self.extract_links(html, current_url)

                            for link_url, link_text in links[:10]:
                                if link_url not in self.visited_urls and len(queue) < 100:
                                    queue.append((link_url, link_text, depth + 1))
                        except Exception:
                            pass

                time.sleep(0.5)

            browser.close()
            return all_content

    # ---------------------------------------------------------------------
    # Markdown output
    # ---------------------------------------------------------------------
    def save_to_markdown(self, content_list, filename):
        """Save scraped content as a combined Markdown file."""
        if not content_list:
            print("  No content to save")
            return

        markdown = f"""# Singapore Travel Guide

**Source:** Scraped from Visit Singapore and Wikivoyage
**Generated:** {str(__import__('datetime').datetime.now())}
**Total Pages:** {len(content_list)}

---

## Table of Contents

"""

        for i, item in enumerate(content_list, 1):
            title_safe = item["title"].replace("#", "").replace("|", "-")[:50]
            markdown += f"{i}. {title_safe}\n"

        markdown += "\n---\n\n"

        for item in content_list:
            try:
                markdown += f"## {item['title']}\n\n"
                markdown += f"**Source:** [{item['url']}]({item['url']})\n\n"
                markdown += item["content"]
                markdown += "\n\n---\n\n"
            except Exception:
                pass

        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)

        file_size = os.path.getsize(filepath) / 1024
        print(f"  Saved: {filepath} ({file_size:.0f} KB, {len(content_list)} pages)")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main():
    """Main script entry point."""
    print("\n" + "=" * 70)
    print("DEEP CONTEXT SCRAPER - SINGAPORE TRAVEL GUIDE")
    print("=" * 70)
    print("\nThis scraper will:")
    print("  1. Load a starting page")
    print("  2. Extract all internal links")
    print("  3. Follow each link and scrape content")
    print("  4. Combine all content into comprehensive markdown")
    print("\nEstimated time: 20-40 minutes")
    print("Internet: Required")
    print("=" * 70 + "\n")

    try:
        scraper = DeepScraperSingapore()

        # 1. Scrape things to do attractions
        print("Scraping: Things to Do")
        print("-" * 70)
        content = scraper.scrape_deep(
            "https://www.visitsingapore.com/things-to-do/",
            max_pages=50,
            max_depth=2,
        )
        scraper.save_to_markdown(content, "things_to_do_complete.md")

        time.sleep(2)

        # 2. Scrape itineraries
        scraper = DeepScraperSingapore()
        print("\nScraping: Itineraries")
        print("-" * 70)
        content = scraper.scrape_deep(
            "https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/",
            max_pages=30,
            max_depth=2,
        )
        scraper.save_to_markdown(content, "itineraries_complete.md")

        time.sleep(2)

        # 3. Scrape neighbourhood information
        scraper = DeepScraperSingapore()
        print("\nScraping: Neighbourhoods")
        print("-" * 70)
        content = scraper.scrape_deep(
            "https://www.visitsingapore.com/neighbourhood/featured-neighbourhood/",
            max_pages=20,
            max_depth=1,
        )
        scraper.save_to_markdown(content, "neighbourhoods_complete.md")

        time.sleep(2)

        # 4. Scrape Wikivoyage content
        scraper = DeepScraperSingapore()
        print("\nScraping: Wikivoyage Singapore")
        print("-" * 70)
        content = scraper.scrape_deep(
            "https://en.wikivoyage.org/wiki/Singapore",
            max_pages=25,
            max_depth=2,
        )
        scraper.save_to_markdown(content, "wikivoyage_complete.md")

        print("\n" + "=" * 70)
        print("SCRAPING COMPLETE")
        print("=" * 70)
        print("\nGenerated files in data/:")
        print("  - things_to_do_complete.md")
        print("  - itineraries_complete.md")
        print("  - neighbourhoods_complete.md")
        print("  - wikivoyage_complete.md")
        print("\nNext steps:")
        print("  1. python knowledge_base_builder.py")
        print("  2. python travel_assistant.py")
        print("\nYour RAG will now have complete information.")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
