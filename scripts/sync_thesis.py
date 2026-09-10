"""
Full Synchronization Script:
Copies docs/thesis/thesis_final_draft.md to docs/thesis/thesis_readable.md and docs/thesis/thesis_readable.txt.
"""

import os

def sync_markdown_and_text():
    thesis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "thesis")
    source = os.path.join(thesis_dir, "thesis_final_draft.md")
    
    if not os.path.exists(source):
        print(f"Error: {source} not found.")
        return

    with open(source, "r", encoding="utf-8") as f:
        content = f.read()

    # Sync to thesis_readable.md and thesis_readable.txt inside docs/thesis
    with open(os.path.join(thesis_dir, "thesis_readable.md"), "w", encoding="utf-8") as f:
        f.write(content)
    with open(os.path.join(thesis_dir, "thesis_readable.txt"), "w", encoding="utf-8") as f:
        f.write(content)

    print("Successfully synchronized thesis_final_draft.md -> thesis_readable.md and thesis_readable.txt inside docs/thesis!")

if __name__ == "__main__":
    sync_markdown_and_text()
