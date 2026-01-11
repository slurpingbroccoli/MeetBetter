from playwright.sync_api import sync_playwright
import time
import os
import json
import re

OUTPUT_JSON = "linkedin_profile.json"
CHROME_USER_DATA_DIR = os.path.expanduser("~/playwright-chrome-profile")

def clean_text(text):
    if not text:
        return None
    return re.sub(r"\s+", " ", text.strip())

def scroll_page(page):
    # Scroll the main window slowly to force LinkedIn to render all sections
    scroll_height = page.evaluate("() => document.body.scrollHeight")
    current = 0
    step = 1000
    while current < scroll_height:
        page.evaluate(f"window.scrollBy(0, {step});")
        time.sleep(0.2)  # wait for virtualized content to render
        current += step
        scroll_height = page.evaluate("() => document.body.scrollHeight")  # update if page grows

def extract_section_content(card, section_name):
    """Extract section content, handling collapsible text and various section types"""
    try:
        section_lower = section_name.lower()
        
        # For Experience/Education/Projects sections with structured lists
        if section_lower in ["experience", "education", "projects"]:
            # Look for the main list container
            main_list = card.locator("ul.GazyoHFORBFeTorpVaslmVyvZbxxSlYAaPz, ul.pvs-list")
            if main_list.count() > 0:
                # Get all list items (top-level entries)
                list_items = main_list.first.locator("li.artdeco-list__item, li.pvs-list__item")
                if list_items.count() > 0:
                    texts = []
                    for j in range(list_items.count()):
                        item_text = clean_text(list_items.nth(j).inner_text())
                        if item_text:
                            texts.append(item_text)
                    if texts:
                        return "\n---\n".join(texts)  # Use separator to distinguish entries
        
        # For About section: try to find the content in the inline-show-more-text div
        content_div = card.locator("div[class*='inline-show-more-text']")
        if content_div.count() > 0:
            # Get text from span with aria-hidden="true" inside the content div
            text_span = content_div.first.locator("span[aria-hidden='true']")
            if text_span.count() > 0:
                text = clean_text(text_span.first.inner_text())
                if text:
                    return text
        
        # Try to get text from the display-flex ph5 pv3 div (content container for About)
        content_area = card.locator("div.display-flex.ph5.pv3")
        if content_area.count() > 0:
            # Get text but exclude button/interaction elements
            full_width = content_area.first.locator("div.full-width")
            if full_width.count() > 0:
                text = clean_text(full_width.first.inner_text())
                if text:
                    return text
            # If that doesn't work, get all text from content area
            text = clean_text(content_area.first.inner_text())
            if text:
                return text
        
        # Fallback: try to get content from profile-component-entity divs
        entity_divs = card.locator("div[data-view-name='profile-component-entity']")
        if entity_divs.count() > 0:
            texts = []
            for j in range(entity_divs.count()):
                item_text = clean_text(entity_divs.nth(j).inner_text())
                if item_text:
                    texts.append(item_text)
            if texts:
                return "\n---\n".join(texts)
        
        # Final fallback: get all text from card (excluding header and buttons)
        header = card.locator("h2")
        full_text = clean_text(card.inner_text())
        if header.count() > 0:
            header_text = clean_text(header.first.inner_text())
            if full_text and header_text and full_text.startswith(header_text):
                # Remove header text
                remaining = clean_text(full_text[len(header_text):])
                # Remove common button text like "Edit about", "Edit experience", etc.
                remaining = re.sub(r'\s*Edit\s+\w+\s*', '', remaining, flags=re.IGNORECASE)
                return remaining if remaining else None
        
        return full_text if full_text else None
    except Exception:
        return None

def extract_profile_cards(page):
    cards_data = {}
    # Select all sections by stable card class (including mt2 variant)
    cards = page.locator("section.artdeco-card.pv-profile-card.break-words")
    for i in range(cards.count()):
        card = cards.nth(i)
        try:
            # Try different header selectors
            header = card.locator("h2.pvs-header__title span[aria-hidden='true']")
            if header.count() == 0:
                header = card.locator("h2 span[aria-hidden='true']")
            
            section_name = clean_text(header.first.inner_text()) if header.count() > 0 else f"section_{i+1}"
            
            # Extract content for all sections
            text_content = extract_section_content(card, section_name)
            
            if text_content:  # Only add if we got content
                cards_data[section_name.lower()] = {
                    "class": card.get_attribute("class"),
                    "text": text_content
                }
        except Exception as e:
            continue
    return cards_data

def scrape_linkedin_profile(url):
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=CHROME_USER_DATA_DIR,
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )

        page = context.new_page()
        page.goto(url, timeout=60000)
        time.sleep(1.5)

        scroll_page(page)  # scroll slowly to render all sections

        profile_data = {
            "name": clean_text(page.locator("h1").first.inner_text()),
            "headline": clean_text(page.locator("div.text-body-medium").first.inner_text()),
            "location": clean_text(page.locator("span.text-body-small.inline.t-black--light.break-words").first.inner_text()),
            "sections": extract_profile_cards(page)
        }

        context.close()

    return profile_data

if __name__ == "__main__":
    scrape_linkedin_profile()