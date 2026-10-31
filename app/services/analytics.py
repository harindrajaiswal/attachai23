"""Real database-backed analytics and usage telemetry service.

Fixes Defect #18:
Computes all dashboard charts, daily conversation/message/lead time-series,
latency metrics, and tenant-level breakdowns from actual persisted database rows.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional
from app.repositories.base import BaseRepository


class AnalyticsService:
    """Aggregates real tenant and platform analytics from persisted database records."""

    def __init__(self, repository: BaseRepository) -> None:
        self.repository = repository

    async def get_dashboard_analytics(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive platform and tenant analytics from real DB tables."""
        clients = await self.repository.list_clients()
        client_map = {c["id"]: c for c in clients}

        conversations = await self.repository.list_conversations(tenant_id=tenant_id, limit=1000)
        leads = await self.repository.list_leads(tenant_id=tenant_id, limit=1000)
        files = await self.repository.list_knowledge_files(tenant_id=tenant_id)
        usage_logs = await self.repository.list_usage_logs(
            tenant_id=tenant_id, start_date=start_date, end_date=end_date
        )

        # Date filter helper
        def in_date_range(iso_ts: Optional[str]) -> bool:
            if not iso_ts:
                return True
            day = str(iso_ts)[:10]
            if start_date and day < start_date[:10]:
                return False
            if end_date and day > end_date[:10]:
                return False
            return True

        filtered_convs = [c for c in conversations if in_date_range(c.get("created_at"))]
        filtered_leads = [l for l in leads if in_date_range(l.get("created_at"))]

        # Daily time-series buckets
        daily_map: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "date": "",
                "conversations": 0,
                "messages": 0,
                "leads": 0,
                "usage_units": 0,
                "avg_latency_ms": 0,
                "_latencies": [],
            }
        )

        for c in filtered_convs:
            day = str(c.get("created_at") or "")[:10] or "unknown"
            daily_map[day]["date"] = day
            daily_map[day]["conversations"] += 1
            daily_map[day]["messages"] += int(c.get("message_count", 0))

        for l in filtered_leads:
            day = str(l.get("created_at") or "")[:10] or "unknown"
            daily_map[day]["date"] = day
            daily_map[day]["leads"] += 1

        latencies_all: List[int] = []
        support_counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNKNOWN": 0}
        total_usage_units = 0
        total_requests = 0

        for u in usage_logs:
            day = str(u.get("created_at") or "")[:10] or "unknown"
            daily_map[day]["date"] = day
            units = int(u.get("usage_units", 1))
            reqs = int(u.get("request_count", 1))
            lat = int(u.get("latency_ms", 0))
            sup = str(u.get("support_level") or "UNKNOWN").upper()

            daily_map[day]["usage_units"] += units
            if lat > 0:
                daily_map[day]["_latencies"].append(lat)
                latencies_all.append(lat)
            total_usage_units += units
            total_requests += reqs
            if sup in support_counts:
                support_counts[sup] += 1

        daily_series: List[Dict[str, Any]] = []
        for day in sorted(daily_map.keys()):
            entry = daily_map[day]
            lats = entry.pop("_latencies", [])
            entry["avg_latency_ms"] = int(sum(lats) / len(lats)) if lats else 0
            daily_series.append(entry)

        avg_latency = int(sum(latencies_all) / len(latencies_all)) if latencies_all else 0
        sorted_lats = sorted(latencies_all)
        p95_latency = sorted_lats[int(len(sorted_lats) * 0.95)] if sorted_lats else 0

        # Per-tenant breakdown
        tenant_stats: List[Dict[str, Any]] = []
        target_clients = [client_map[tenant_id]] if (tenant_id and tenant_id in client_map) else clients
        for cl in target_clients:
            cid = cl["id"]
            t_convs = [c for c in filtered_convs if c.get("tenant_id") == cid]
            t_leads = [l for l in filtered_leads if l.get("tenant_id") == cid]
            t_files = [f for f in files if f.get("tenant_id") == cid]
            t_usage = [u for u in usage_logs if u.get("tenant_id") == cid]
            t_lats = [int(u.get("latency_ms", 0)) for u in t_usage if int(u.get("latency_ms", 0)) > 0]
            tenant_stats.append(
                {
                    "tenant_id": cid,
                    "business_name": cl["business_name"],
                    "slug": cl["slug"],
                    "enabled": cl["enabled"],
                    "conversations": len(t_convs),
                    "messages": sum(int(c.get("message_count", 0)) for c in t_convs),
                    "leads": len(t_leads),
                    "knowledge_files": len(t_files),
                    "knowledge_chunks": sum(int(f.get("chunk_count", 0)) for f in t_files),
                    "usage_units": sum(int(u.get("usage_units", 0)) for u in t_usage),
                    "avg_latency_ms": int(sum(t_lats) / len(t_lats)) if t_lats else 0,
                }
            )

        total_messages = sum(int(c.get("message_count", 0)) for c in filtered_convs)
        total_chunks = sum(int(f.get("chunk_count", 0)) for f in files if f.get("status") == "ready")

        return {
            "summary": {
                "total_tenants": len(clients),
                "active_tenants": sum(1 for c in clients if c.get("enabled")),
                "total_conversations": len(filtered_convs),
                "total_messages": total_messages,
                "total_leads": len(filtered_leads),
                "total_knowledge_files": len(files),
                "ready_knowledge_files": sum(1 for f in files if f.get("status") == "ready"),
                "total_knowledge_chunks": total_chunks,
                "total_requests": total_requests,
                "total_usage_units": total_usage_units,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "support_breakdown": support_counts,
            },
            "daily_series": daily_series,
            "tenant_stats": tenant_stats,
        }
