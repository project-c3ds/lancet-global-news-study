from eventregistry import *
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('NEWSAPI_KEY')
er = EventRegistry(apiKey=api_key)

CLIMATE_KEYWORDS_EN = [
    "climate change", "climate changes", "changing climate", "changing climates",
    "environmental change", "environmental changes",
    "global warming", "global heating",
    "climate emergency", "climate emergencies",
    "climate crisis", "climate crises",
    "climate action", "climate actions",
    "climate variability", "variable climate", "variable climates",
    "extreme weather", "extreme event", "extreme events",
    "extreme heat", "heatwave", "heatwaves",
    "rising temperature", "rising temperatures",
    "temperature rise", "temperature rises",
    "sea level rise", "rising sea level", "rising sea levels",
    "greenhouse gas", "greenhouse gases",
    "carbon emission", "carbon emissions",
    "co2 emission", "co2 emissions", "carbon dioxide",
]


def fetch_source_uri(url):
    """Fetch the source URI from EventRegistry for a given URL."""
    return er.getSourceUri(url)


def collect_recent_articles(source_uri, max_items=20,
                            date_start='2025-03-01', date_end='2026-03-06'):
    """Collect the most recent articles from a source (no keyword filter).
    Returns a dict with source metadata and the raw articles.
    Uses a 1-year date range to minimize token cost (5 tokens/year searched)."""
    q = QueryArticles(
        sourceUri=source_uri,
        dateStart=date_start,
        dateEnd=date_end,
    )
    q.setRequestedResult(RequestArticlesInfo(
        count=max_items,
        sortBy="date",
        sortByAsc=False,
        returnInfo=ReturnInfo(
            articleInfo=ArticleInfoFlags(body=False, image=False,
                                        sentiment=False, authors=False)
        )
    ))
    res = er.execQuery(q)
    articles = res.get("articles", {}).get("results", [])

    # Extract metadata from collected articles
    lang_counts = Counter(a.get("lang", "unknown") for a in articles)
    most_recent_date = None
    if articles:
        dates = [a.get("dateTime") for a in articles if a.get("dateTime")]
        if dates:
            most_recent_date = max(dates)

    return {
        "articles_collected": len(articles),
        "articles_requested": max_items,
        "most_recent_date": most_recent_date,
        "language_counts": dict(lang_counts),
        "most_frequent_language": lang_counts.most_common(1)[0][0] if lang_counts else None,
        "articles": articles,
    }


def count_articles(source_uri, keywords=None,
                   date_start='2020-01-01', date_end='2026-03-06'):
    """Count articles matching keywords for a source without fetching them.
    Keywords should be a list of phrases -- they are OR'd together via QueryItems.OR()."""
    if keywords is None:
        keywords = CLIMATE_KEYWORDS_EN
    q = QueryArticlesIter(
        keywords=QueryItems.OR(keywords),
        sourceUri=source_uri,
        dateStart=date_start,
        dateEnd=date_end,
        keywordSearchMode='phrase'
    )
    return q.count(er)
