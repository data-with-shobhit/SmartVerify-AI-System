from collections import defaultdict
from core.logger import get_logger

logger = get_logger("reconciler")

class OrderReconciler:
    def reconcile(self, expected_order: list, verifications: list) -> dict:
        """
        Compare physical detections (verifications) against the customer's bill.
        """
        # Count what we found
        detected_counts = defaultdict(list)
        for det in verifications:
            if det["status"] in ("VERIFIED", "REVIEW"):
                detected_counts[det["verified_sku_id"]].append(det)

        logger.info(f"Reconciling order. Total raw detections: {len(verifications)}")

        order_lines = []
        overall_status = "VERIFIED"
        summary = {"verified": 0, "review": 0, "missing": 0, "mismatch": 0}

        for item in expected_order:
            sku_id = item["sku_id"]
            expected_qty = item.get("qty", 1)
            found_items = detected_counts.get(sku_id, [])
            found_qty = len(found_items)

            # Check for REVIEW status in the found items
            needs_review = any(d["status"] == "REVIEW" for d in found_items)
            
            status = "VERIFIED"
            if found_qty == 0:
                status = "MISSING"
                overall_status = "ISSUE"
                summary["missing"] += 1
            elif found_qty < expected_qty:
                status = "PARTIAL"
                overall_status = "ISSUE"
                summary["missing"] += 1
            elif found_qty > expected_qty:
                status = "REVIEW" # Excess items
                overall_status = "ISSUE"
                summary["review"] += 1
            elif needs_review:
                status = "REVIEW"
                overall_status = "ISSUE"
                summary["review"] += 1
            else:
                summary["verified"] += 1

            logger.info(f" -> Result for {item['product_name']}: Expected {expected_qty}, Found {found_qty} -> Status: {status}")

            order_line = {
                "sku_id": sku_id,
                "product_name": item["product_name"],
                "expected": expected_qty,
                "found": found_qty,
                "status": status,
                "items": found_items
            }
            order_lines.append(order_line)

        return {
            "overall_status": overall_status,
            "summary": summary,
            "order_lines": order_lines
        }
