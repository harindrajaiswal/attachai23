"""Public widget configuration, theme settings, and static asset serving tests."""

import unittest
from pathlib import Path
from tests.conftest import IsolatedOrbitTestEnv


class TestWidget(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IsolatedOrbitTestEnv()
        self.tenant, self.api_key = self.env.create_test_tenant()

    def tearDown(self) -> None:
        self.env.cleanup()

    def test_widget_config_reflects_theme_updates_without_leaking_secrets(self) -> None:
        # Update widget settings via admin API
        self.env.client.patch(
            f"/api/admin/clients/{self.tenant['id']}",
            json_body={
                "assistant_name": "Nova Concierge",
                "welcome_message": "Welcome to Nova!",
                "widget_config": {
                    "primary_color": "#0d9488",
                    "theme_mode": "dark",
                    "position": "bottom-left",
                    "width": 410,
                    "height": 600,
                    "border_radius": 18,
                    "suggested_questions": ["Question A?", "Question B?"],
                },
            },
            headers=self.env.admin_headers,
        )

        cfg_resp = self.env.client.get("/v1/config", headers={"x-api-key": self.api_key})
        self.assertEqual(cfg_resp.status_code, 200)
        cfg = cfg_resp.json()
        self.assertEqual(cfg["assistantName"], "Nova Concierge")
        self.assertEqual(cfg["welcomeMessage"], "Welcome to Nova!")
        self.assertEqual(cfg["theme"]["primaryColor"], "#0d9488")
        self.assertEqual(cfg["theme"]["themeMode"], "dark")
        self.assertEqual(cfg["theme"]["position"], "bottom-left")
        self.assertEqual(cfg["theme"]["width"], 410)
        self.assertEqual(cfg["theme"]["borderRadius"], 18)

        # Ensure no private prompts, key hashes, or secrets appear in /v1/config
        raw_json = cfg_resp.text.lower()
        self.assertNotIn("system_prompt", raw_json)
        self.assertNotIn("key_hash", raw_json)
        self.assertNotIn("admin_api_key", raw_json)

    def test_widget_js_declares_lead_submitted_state_cleanly(self) -> None:
        js_resp = self.env.client.get("/static/js/widget.js")
        self.assertEqual(js_resp.status_code, 200)
        js_code = js_resp.text
        self.assertIn("leadSubmitted: false", js_code)
        self.assertIn("state.leadSubmitted", js_code)


if __name__ == "__main__":
    unittest.main()
