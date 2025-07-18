# utils/markdown.py (Enhanced for better table/code rendering, inspired by repo)
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.nl2br import Nl2BrExtension

def render_markdown(text: str) -> str:
    """
    Convert text to HTML markdown with extensions for tables, code, line breaks.
    """
    if not text:
        return ""
    extensions = [TableExtension(), FencedCodeExtension(), Nl2BrExtension()]
    html = markdown.markdown(text, extensions=extensions)
    # Add custom styling if needed
    styled_html = f'<div style="font-family: Arial, sans-serif; line-height: 1.6;">{html}</div>'
    return styled_html
