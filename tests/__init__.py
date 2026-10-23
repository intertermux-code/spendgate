"""spendgate's test suite: stdlib unittest only, no third-party runners.

Imported before any test module, so the proxy's own warning logs (budget alerts
are deliberately loud) do not bury the test results.
"""

import logging

logging.getLogger("spendgate").setLevel(logging.CRITICAL)
