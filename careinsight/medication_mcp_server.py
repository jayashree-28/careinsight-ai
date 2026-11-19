import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer


mcp = MCPServer(
    name="CareInsight Medication Information Server",
    instructions=(
        "Provides medication labeling information "
        "from the US FDA openFDA database."
    ),
)


@mcp.tool()
def lookup_medication_info(name: str) -> str:
    """
    Look up FDA medication labeling information by brand or generic name.

    Args:
        name: Medication brand name or generic name.

    Returns:
        Relevant medication labeling information from openFDA.
    """

    if not name or not name.strip():
        return "Please provide a medication name."

    medication = name.strip()

    search_query = (
        f'openfda.brand_name:"{medication}" OR '
        f'openfda.generic_name:"{medication}"'
    )

    params = urllib.parse.urlencode({
        "search": search_query,
        "limit": 1,
    })

    url = f"https://api.fda.gov/drug/label.json?{params}"

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CareInsightAI/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = data.get("results", [])

        if not results:
            return f"No FDA labeling information found for {medication}."

        label = results[0]
        openfda = label.get("openfda", {})

        result = {
            "medication_searched": medication,
            "brand_name": openfda.get("brand_name", []),
            "generic_name": openfda.get("generic_name", []),
            "manufacturer": openfda.get("manufacturer_name", []),
            "indications_and_usage": label.get(
                "indications_and_usage", []
            ),
            "warnings": label.get("warnings", []),
            "adverse_reactions": label.get(
                "adverse_reactions", []
            ),
            "drug_interactions": label.get(
                "drug_interactions", []
            ),
            "source": "US FDA openFDA Drug Label API",
            "source_url": "https://open.fda.gov/apis/drug/label/",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "disclaimer": (
                "Labeling information is provided for reference. "
                "It does not establish that a medication caused "
                "a reported patient experience and is not medical advice."
            ),
        }

        return json.dumps(result, indent=2)

    except urllib.error.HTTPError as error:
        if error.code == 404:
            return f"No FDA labeling information found for {medication}."

        return f"FDA API returned HTTP error {error.code}."

    except urllib.error.URLError:
        return "Unable to connect to the FDA API. Please try again."

    except (TimeoutError, json.JSONDecodeError):
        return "The FDA API request timed out or returned invalid data."


if __name__ == "__main__":
    mcp.run(transport="stdio")
