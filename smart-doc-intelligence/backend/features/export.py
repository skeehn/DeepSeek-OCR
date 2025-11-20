"""
Export Module
Exports analysis results to various formats: JSON, Markdown, PDF, HTML
"""
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExportMetadata:
    """Metadata for exports"""
    export_format: str
    export_date: str
    content_type: str
    source_document: Optional[str] = None
    generated_by: str = "Smart Document Intelligence Platform"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class JSONExporter:
    """
    Exports data to JSON format
    """

    def __init__(self, pretty: bool = True):
        """
        Initialize JSON exporter

        Args:
            pretty: Pretty-print JSON
        """
        self.pretty = pretty

        print(f"✅ JSON Exporter initialized")

    def export(
        self,
        data: Dict[str, Any],
        output_path: str,
        include_metadata: bool = True
    ) -> str:
        """
        Export data to JSON file

        Args:
            data: Data to export
            output_path: Output file path
            include_metadata: Include export metadata

        Returns:
            Output file path
        """
        print(f"\n💾 Exporting to JSON: {output_path}")

        # Add metadata
        if include_metadata:
            export_data = {
                "metadata": ExportMetadata(
                    export_format="json",
                    export_date=datetime.now().isoformat(),
                    content_type=data.get("type", "unknown")
                ).to_dict(),
                "data": data
            }
        else:
            export_data = data

        # Write JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if self.pretty:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(export_data, f, ensure_ascii=False)

        print(f"✅ Exported to {output_path}")

        return output_path


class MarkdownExporter:
    """
    Exports data to Markdown format
    """

    def __init__(self):
        """Initialize Markdown exporter"""
        print(f"✅ Markdown Exporter initialized")

    def export(
        self,
        data: Dict[str, Any],
        output_path: str,
        title: Optional[str] = None
    ) -> str:
        """
        Export data to Markdown file

        Args:
            data: Data to export
            output_path: Output file path
            title: Document title

        Returns:
            Output file path
        """
        print(f"\n📝 Exporting to Markdown: {output_path}")

        # Build markdown content
        content_type = data.get("type", "document")

        if content_type == "summary":
            markdown = self._export_summary(data, title)
        elif content_type == "comparison":
            markdown = self._export_comparison(data, title)
        elif content_type == "entities":
            markdown = self._export_entities(data, title)
        elif content_type == "rag_response":
            markdown = self._export_rag_response(data, title)
        else:
            markdown = self._export_generic(data, title)

        # Write file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"✅ Exported to {output_path}")

        return output_path

    def _export_summary(self, data: Dict[str, Any], title: Optional[str]) -> str:
        """Export summary to Markdown"""
        md = []

        # Header
        md.append(f"# {title or 'Document Summary'}\n")
        md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Summary
        if "summary" in data:
            md.append("## Summary\n")
            md.append(f"{data['summary']}\n")

        # Key points
        if "key_points" in data:
            md.append("## Key Points\n")
            for point in data["key_points"]:
                md.append(f"- {point}")
            md.append("")

        # Metadata
        if "metadata" in data:
            md.append("## Metadata\n")
            for key, value in data["metadata"].items():
                md.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            md.append("")

        return "\n".join(md)

    def _export_comparison(self, data: Dict[str, Any], title: Optional[str]) -> str:
        """Export comparison to Markdown"""
        md = []

        # Header
        md.append(f"# {title or 'Document Comparison'}\n")
        md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Documents
        if "doc_ids" in data:
            md.append("## Documents Compared\n")
            for doc_id in data["doc_ids"]:
                md.append(f"- {doc_id}")
            md.append("")

        # Similarity scores
        if "similarity_scores" in data:
            md.append("## Similarity Scores\n")
            for pair, score in data["similarity_scores"].items():
                md.append(f"- **{pair}**: {score:.1%}")
            md.append("")

        # Common terms
        if "common_terms" in data:
            md.append("## Common Terms\n")
            for item in data["common_terms"][:15]:
                if isinstance(item, dict):
                    md.append(f"- {item['term']} ({item['count']} occurrences)")
                else:
                    md.append(f"- {item}")
            md.append("")

        # Summary
        if "summary" in data:
            md.append("## Analysis Summary\n")
            md.append(f"{data['summary']}\n")

        return "\n".join(md)

    def _export_entities(self, data: Dict[str, Any], title: Optional[str]) -> str:
        """Export entity extraction to Markdown"""
        md = []

        # Header
        md.append(f"# {title or 'Entity Extraction'}\n")
        md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Entities by type
        if "entities" in data:
            md.append("## Extracted Entities\n")

            for entity_type, entities in data["entities"].items():
                md.append(f"### {entity_type.upper()}\n")

                for entity in entities[:10]:  # Limit to top 10
                    if isinstance(entity, dict):
                        text = entity.get("text", "")
                        count = entity.get("occurrences", 0)
                        confidence = entity.get("confidence", 0)
                        md.append(f"- **{text}** (occurrences: {count}, confidence: {confidence:.0%})")
                    else:
                        md.append(f"- {entity}")

                md.append("")

        # Key terms
        if "key_terms" in data:
            md.append("## Key Terms\n")
            for item in data["key_terms"][:20]:
                if isinstance(item, dict):
                    md.append(f"- {item['term']}: {item['frequency']}")
                else:
                    md.append(f"- {item}")
            md.append("")

        return "\n".join(md)

    def _export_rag_response(self, data: Dict[str, Any], title: Optional[str]) -> str:
        """Export RAG response to Markdown"""
        md = []

        # Header
        md.append(f"# {title or 'Query Response'}\n")
        md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Query
        if "query" in data:
            md.append("## Query\n")
            md.append(f"> {data['query']}\n")

        # Answer
        if "answer" in data:
            md.append("## Answer\n")
            md.append(f"{data['answer']}\n")

        # Sources
        if "sources" in data and data["sources"]:
            md.append("## Sources\n")
            for i, source in enumerate(data["sources"][:5], 1):
                if isinstance(source, dict):
                    doc_id = source.get("doc_id", "Unknown")
                    text = source.get("text", "")[:200]
                    score = source.get("score", 0)
                    md.append(f"### Source {i} (Relevance: {score:.2f})")
                    md.append(f"**Document**: {doc_id}")
                    md.append(f"```\n{text}...\n```\n")

        # Metadata
        if "metadata" in data:
            md.append("## Metadata\n")
            for key, value in data["metadata"].items():
                md.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            md.append("")

        return "\n".join(md)

    def _export_generic(self, data: Dict[str, Any], title: Optional[str]) -> str:
        """Export generic data to Markdown"""
        md = []

        # Header
        md.append(f"# {title or 'Document Export'}\n")
        md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Content
        md.append("## Content\n")

        for key, value in data.items():
            if key == "type":
                continue

            md.append(f"### {key.replace('_', ' ').title()}\n")

            if isinstance(value, str):
                md.append(f"{value}\n")
            elif isinstance(value, (list, dict)):
                md.append(f"```json\n{json.dumps(value, indent=2)}\n```\n")
            else:
                md.append(f"{value}\n")

        return "\n".join(md)


class HTMLExporter:
    """
    Exports data to HTML format
    """

    def __init__(self):
        """Initialize HTML exporter"""
        print(f"✅ HTML Exporter initialized")

    def export(
        self,
        data: Dict[str, Any],
        output_path: str,
        title: Optional[str] = None,
        include_css: bool = True
    ) -> str:
        """
        Export data to HTML file

        Args:
            data: Data to export
            output_path: Output file path
            title: Document title
            include_css: Include CSS styling

        Returns:
            Output file path
        """
        print(f"\n🌐 Exporting to HTML: {output_path}")

        # Build HTML
        html = self._build_html(data, title or "Document Export", include_css)

        # Write file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✅ Exported to {output_path}")

        return output_path

    def _build_html(self, data: Dict[str, Any], title: str, include_css: bool) -> str:
        """Build HTML document"""
        css = self._get_css() if include_css else ""

        html_body = self._build_body(data)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {css}
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {html_body}
    </div>
</body>
</html>"""

        return html

    def _build_body(self, data: Dict[str, Any]) -> str:
        """Build HTML body content"""
        content_type = data.get("type", "generic")

        if content_type == "summary":
            return self._build_summary_html(data)
        elif content_type == "comparison":
            return self._build_comparison_html(data)
        elif content_type == "entities":
            return self._build_entities_html(data)
        elif content_type == "rag_response":
            return self._build_rag_html(data)
        else:
            return self._build_generic_html(data)

    def _build_summary_html(self, data: Dict[str, Any]) -> str:
        """Build HTML for summary"""
        html = []

        if "summary" in data:
            html.append(f'<div class="section">')
            html.append(f'<h2>Summary</h2>')
            html.append(f'<p>{data["summary"]}</p>')
            html.append(f'</div>')

        if "key_points" in data:
            html.append(f'<div class="section">')
            html.append(f'<h2>Key Points</h2>')
            html.append(f'<ul>')
            for point in data["key_points"]:
                html.append(f'<li>{point}</li>')
            html.append(f'</ul>')
            html.append(f'</div>')

        return "\n".join(html)

    def _build_comparison_html(self, data: Dict[str, Any]) -> str:
        """Build HTML for comparison"""
        html = []

        if "similarity_scores" in data:
            html.append(f'<div class="section">')
            html.append(f'<h2>Similarity Scores</h2>')
            html.append(f'<ul>')
            for pair, score in data["similarity_scores"].items():
                html.append(f'<li><strong>{pair}</strong>: {score:.1%}</li>')
            html.append(f'</ul>')
            html.append(f'</div>')

        if "summary" in data:
            html.append(f'<div class="section">')
            html.append(f'<h2>Analysis</h2>')
            html.append(f'<p>{data["summary"]}</p>')
            html.append(f'</div>')

        return "\n".join(html)

    def _build_entities_html(self, data: Dict[str, Any]) -> str:
        """Build HTML for entities"""
        html = []

        if "entities" in data:
            html.append(f'<div class="section">')
            html.append(f'<h2>Extracted Entities</h2>')

            for entity_type, entities in data["entities"].items():
                html.append(f'<h3>{entity_type.upper()}</h3>')
                html.append(f'<ul>')
                for entity in entities[:10]:
                    if isinstance(entity, dict):
                        text = entity.get("text", "")
                        count = entity.get("occurrences", 0)
                        html.append(f'<li><strong>{text}</strong> ({count} occurrences)</li>')
                html.append(f'</ul>')

            html.append(f'</div>')

        return "\n".join(html)

    def _build_rag_html(self, data: Dict[str, Any]) -> str:
        """Build HTML for RAG response"""
        html = []

        if "query" in data:
            html.append(f'<div class="section query">')
            html.append(f'<h2>Query</h2>')
            html.append(f'<p class="query-text">{data["query"]}</p>')
            html.append(f'</div>')

        if "answer" in data:
            html.append(f'<div class="section answer">')
            html.append(f'<h2>Answer</h2>')
            html.append(f'<p>{data["answer"]}</p>')
            html.append(f'</div>')

        if "sources" in data and data["sources"]:
            html.append(f'<div class="section">')
            html.append(f'<h2>Sources</h2>')
            for i, source in enumerate(data["sources"][:5], 1):
                if isinstance(source, dict):
                    doc_id = source.get("doc_id", "Unknown")
                    text = source.get("text", "")[:200]
                    html.append(f'<div class="source">')
                    html.append(f'<h3>Source {i}</h3>')
                    html.append(f'<p><strong>Document:</strong> {doc_id}</p>')
                    html.append(f'<pre>{text}...</pre>')
                    html.append(f'</div>')
            html.append(f'</div>')

        return "\n".join(html)

    def _build_generic_html(self, data: Dict[str, Any]) -> str:
        """Build HTML for generic data"""
        html = ['<div class="section">']

        for key, value in data.items():
            if key == "type":
                continue

            html.append(f'<h2>{key.replace("_", " ").title()}</h2>')

            if isinstance(value, str):
                html.append(f'<p>{value}</p>')
            elif isinstance(value, (list, dict)):
                html.append(f'<pre>{json.dumps(value, indent=2)}</pre>')
            else:
                html.append(f'<p>{value}</p>')

        html.append('</div>')

        return "\n".join(html)

    def _get_css(self) -> str:
        """Get CSS styling"""
        return """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        h3 {
            color: #7f8c8d;
        }
        .timestamp {
            color: #95a5a6;
            font-style: italic;
            margin-bottom: 30px;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }
        .query {
            background-color: #e8f4f8;
            border-left-color: #3498db;
        }
        .answer {
            background-color: #e8f8f0;
            border-left-color: #27ae60;
        }
        .query-text {
            font-size: 1.1em;
            font-weight: 500;
        }
        .source {
            margin: 15px 0;
            padding: 15px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        ul {
            padding-left: 25px;
        }
        li {
            margin: 8px 0;
        }
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        strong {
            color: #2c3e50;
        }
    </style>
        """


class ExportManager:
    """
    High-level export manager
    Handles export to multiple formats
    """

    def __init__(self):
        """Initialize export manager"""
        self.json_exporter = JSONExporter()
        self.markdown_exporter = MarkdownExporter()
        self.html_exporter = HTMLExporter()

        print(f"✅ Export Manager initialized")

    def export(
        self,
        data: Dict[str, Any],
        output_dir: str,
        filename: str,
        formats: List[str] = ["json", "markdown", "html"],
        title: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export data to multiple formats

        Args:
            data: Data to export
            output_dir: Output directory
            filename: Base filename (without extension)
            formats: List of formats to export
            title: Document title

        Returns:
            Dictionary mapping format to output path
        """
        print(f"\n📤 Exporting to multiple formats...")
        print(f"   Output directory: {output_dir}")
        print(f"   Formats: {', '.join(formats)}")

        output_paths = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Export to each format
        if "json" in formats:
            json_path = os.path.join(output_dir, f"{filename}.json")
            output_paths["json"] = self.json_exporter.export(data, json_path)

        if "markdown" in formats or "md" in formats:
            md_path = os.path.join(output_dir, f"{filename}.md")
            output_paths["markdown"] = self.markdown_exporter.export(data, md_path, title)

        if "html" in formats:
            html_path = os.path.join(output_dir, f"{filename}.html")
            output_paths["html"] = self.html_exporter.export(data, html_path, title)

        print(f"\n✅ Export complete!")
        print(f"   Exported {len(output_paths)} files")

        return output_paths


# Convenience functions
def quick_export(
    data: Dict[str, Any],
    output_path: str,
    format: str = "json"
) -> str:
    """Quick export to single format"""
    if format == "json":
        exporter = JSONExporter()
        return exporter.export(data, output_path)
    elif format in ["markdown", "md"]:
        exporter = MarkdownExporter()
        return exporter.export(data, output_path)
    elif format == "html":
        exporter = HTMLExporter()
        return exporter.export(data, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    # Test export
    print("Export Test")
    print("=" * 50)

    # Test data
    test_data = {
        "type": "summary",
        "summary": "This is a test summary of a document about artificial intelligence and machine learning.",
        "key_points": [
            "AI is transforming various industries",
            "Machine learning enables computers to learn from data",
            "Deep learning uses neural networks"
        ],
        "metadata": {
            "word_count": 150,
            "compression_ratio": 0.25
        }
    }

    # Create test exports
    output_dir = "./test_exports"
    manager = ExportManager()

    paths = manager.export(
        test_data,
        output_dir,
        "test_summary",
        formats=["json", "markdown", "html"],
        title="Test Summary"
    )

    print(f"\n✅ Test complete!")
    for format, path in paths.items():
        print(f"   {format}: {path}")
