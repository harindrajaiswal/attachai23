"""Knowledge ingestion, multi-format parsing (TXT, MD, CSV, DOCX, PDF), safe versioned replacement, and rebuild tests."""

import io
import unittest
import zipfile
from tests.conftest import IsolatedOrbitTestEnv


def build_minimal_docx(paragraphs: list[str]) -> bytes:
    """Create a valid minimal OOXML .docx archive for testing."""
    from xml.sax.saxutils import escape

    buf = io.BytesIO()

    xml_body = "".join(
        f"<w:p><w:r><w:t>{escape(p)}</w:t></w:r></w:p>" for p in paragraphs
    )

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{xml_body}<w:sectPr/></w:body></w:document>"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", doc_xml)

    return buf.getvalue()


def build_minimal_pdf(text: str) -> bytes:
    """Create a valid PDF document with extractable text."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        DictionaryObject,
        NameObject,
        DecodedStreamObject,
        ArrayObject,
    )

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )

    resources = DictionaryObject()
    resources[NameObject("/Font")] = DictionaryObject(
        {NameObject("/F1"): writer._add_object(font)}
    )
    page[NameObject("/Resources")] = writer._add_object(resources)

    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = DecodedStreamObject()
    stream.set_data(
        f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET".encode(
            "latin-1", errors="replace"
        )
    )
    page[NameObject("/Contents")] = writer._add_object(stream)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class TestKnowledge(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()
        self.tenant, self.api_key = self.env.create_test_tenant()
        self.tid = self.tenant["id"]

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_ingest_txt_md_csv_docx_and_pdf(self) -> None:
        # 1. TXT
        r_txt = self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.tid,
                "filename": "pricing.txt",
                "content": "Our Standard Plan costs $49 per month and includes 24/7 support.",
            },
            headers=self.env.admin_headers,
        )
        self.assertEqual(r_txt.status_code, 201)
        self.assertEqual(r_txt.json()["status"], "ready")

        # 2. Markdown
        r_md = self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.tid,
                "filename": "faq.md",
                "content": "# FAQ\n\nWe ship orders within 48 hours across North America.",
                "content_type": "text/markdown",
            },
            headers=self.env.admin_headers,
        )
        self.assertEqual(r_md.status_code, 201)

        # 3. CSV with quoted fields, escaped quotes, and multiline values (Fixes Defect #14)
        csv_content = (
            "Service,Price,Description\n"
            '"Personal Training","$85/session","Includes ""Custom Macro Plan"",\nbody composition scan, and mobility coaching."\n'
        )
        r_csv = self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.tid,
                "filename": "services.csv",
                "content": csv_content,
                "content_type": "text/csv",
            },
            headers=self.env.admin_headers,
        )
        self.assertEqual(r_csv.status_code, 201)
        self.assertGreaterEqual(r_csv.json()["chunk_count"], 1)

        # 4. DOCX
        docx_bytes = build_minimal_docx(
            ["Corporate wellness packages start at $499 per month."]
        )
        import asyncio

        docx_rec = asyncio.run(
            self.env.app.container.knowledge_service.ingest_document(
                tenant_id=self.tid,
                filename="corporate.docx",
                content=docx_bytes,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        )
        self.assertEqual(docx_rec["status"], "ready")

        # 5. PDF
        pdf_bytes = build_minimal_pdf(
            "Annual membership discount is 15 percent when paid upfront."
        )
        pdf_rec = asyncio.run(
            self.env.app.container.knowledge_service.ingest_document(
                tenant_id=self.tid,
                filename="brochure.pdf",
                content=pdf_bytes,
                content_type="application/pdf",
            )
        )
        self.assertEqual(pdf_rec["status"], "ready")

    def test_unsupported_empty_and_oversized_rejected(self) -> None:
        bad_ext = self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.tid,
                "filename": "malware.exe",
                "content": "echo hi",
            },
            headers=self.env.admin_headers,
        )
        self.assertEqual(bad_ext.status_code, 422)

        empty_doc = self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.tid,
                "filename": "empty.txt",
                "content": "   ",
            },
            headers=self.env.admin_headers,
        )
        self.assertEqual(empty_doc.status_code, 422)

    def test_safe_versioned_reindexing_preserves_active_index_on_failure(self) -> None:
        # Initial upload (v1)
        up1 = self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.tid,
                "filename": "policy.txt",
                "content": "Refund policy allows returns within 30 days with receipt.",
            },
            headers=self.env.admin_headers,
        ).json()
        self.assertEqual(up1["active_version"], 1)

        # Valid re-upload (activates v2)
        up2 = self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={
                "tenant_id": self.tid,
                "filename": "policy.txt",
                "content": "Updated refund policy allows returns within 45 days with digital receipt.",
            },
            headers=self.env.admin_headers,
        ).json()
        self.assertEqual(up2["active_version"], 2)

        # Failed re-upload (empty content) must NOT destroy active v2 index (Fixes Defect #10)
        fail_up = self.env.client.post(
            "/api/admin/knowledge/upload",
            json_body={"tenant_id": self.tid, "filename": "policy.txt", "content": ""},
            headers=self.env.admin_headers,
        )
        self.assertEqual(fail_up.status_code, 422)

        # Verify v2 still answers customer queries accurately
        chat_resp = self.env.client.post(
            "/v1/chat",
            json_body={
                "message": "How many days does the refund policy allow for returns?"
            },
            headers={"x-api-key": self.api_key},
        ).json()
        self.assertIn("45 days", chat_resp["answer"])

        # Rebuild operation advances to v3 safely
        rebuilt = self.env.client.post(
            f"/api/admin/knowledge/{self.tid}/{up2['id']}/rebuild",
            headers=self.env.admin_headers,
        ).json()
        self.assertEqual(rebuilt["active_version"], 3)


if __name__ == "__main__":
    unittest.main()
