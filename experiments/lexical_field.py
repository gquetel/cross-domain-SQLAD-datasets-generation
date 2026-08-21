"""Contrast the per-domain lexical field across the four domains.

Idea (discussed with Thomas): the four domains share the same SQLIA *task*, so
they share the SQL "protocol" vocabulary (keywords + built-in functions). To
surface how *different* the domains are lexically we strip that shared protocol
vocabulary and look only at what remains -- the domain words (table/column
names, entity names, literal values, ...).

Protocol:
  1. For each domain, pool its in-domain benign train queries (a-a, b-b, c-c,
     d-d; split == train, label == 0), exactly like embedding_viz.py.
  2. Fit one CountVectorizer per domain, with the MySQL keyword + function sets
     (copied from the project constants) passed as stop words so task-specific
     protocol tokens are dropped.
  3. Report the top --top-n most frequent remaining (domain) words per domain.

Outputs (under ../output):
  - lexical_field_top_words.csv : one row per (domain, rank, word, count);
  - lexical_field.pdf           : a 2x2 panel of horizontal bar charts, one
    per domain, top words ranked by frequency.

With --tex-output PATH, the same 2x2 panel is additionally emitted as a pure
pgfplots/TikZ figure (one ``subfigure`` per domain, ``xbar`` charts), so the
figure inherits the document's font settings instead of plotly's rasterised
fonts. This mirrors the tex-figure scripts in mlops-sqldetect/tools (e.g.
generate_drift_figure.py).

Usage:
    python lexical_field.py [--top-n 20] [--n-samples N] \
        [--tex-output ../../quetel_phd_latex/papers/superviz26/data/superviz26-lexical-field.tex]
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer

from diversity_metric import DATASETS_DIR, DTYPES
from embedding_viz import DOMAIN_COLORS, DOMAIN_FILES, load_train_queries


# --- Task-specific "protocol" vocabulary -----------------------------------
# Copied from cross-domain-SQLAD-datasets-generation/models/constants.py. These
# are the MySQL keywords and built-in functions: the SQL vocabulary shared by
# every domain because they all share the SQLIA detection task. We drop them so
# the top words reflect each domain's own lexical field rather than the task.

# https://dev.mysql.com/doc/refman/8.0/en/information-schema-keywords-table.html
mysql_keywords = {
    "ACCESSIBLE", "ADD", "ALL", "ALTER", "ANALYZE", "AND", "AS", "ASC",
    "ASENSITIVE", "BEFORE", "BETWEEN", "BIGINT", "BINARY", "BLOB", "BOTH", "BY",
    "CALL", "CASCADE", "CASE", "CHANGE", "CHAR", "CHARACTER", "CHECK", "COLLATE",
    "COLUMN", "CONDITION", "CONSTRAINT", "CONTINUE", "CONVERT", "CREATE",
    "CROSS", "CUME_DIST", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
    "CURRENT_USER", "CURSOR", "DATABASE", "DATABASES", "DAY_HOUR",
    "DAY_MICROSECOND", "DAY_MINUTE", "DAY_SECOND", "DEC", "DECIMAL", "DECLARE",
    "DEFAULT", "DELAYED", "DELETE", "DENSE_RANK", "DESC", "DESCRIBE",
    "DETERMINISTIC", "DISTINCT", "DISTINCTROW", "DIV", "DOUBLE", "DROP", "DUAL",
    "EACH", "ELSE", "ELSEIF", "EMPTY", "ENCLOSED", "ESCAPED", "EXCEPT", "EXISTS",
    "EXIT", "EXPLAIN", "FALSE", "FETCH", "FIRST_VALUE", "FLOAT", "FLOAT4",
    "FLOAT8", "FOR", "FORCE", "FOREIGN", "FROM", "FULLTEXT", "FUNCTION",
    "GENERATED", "GET", "GRANT", "GROUP", "GROUPING", "GROUPS", "HAVING",
    "HIGH_PRIORITY", "HOUR_MICROSECOND", "HOUR_MINUTE", "HOUR_SECOND", "IF",
    "IGNORE", "IN", "INDEX", "INFILE", "INNER", "INOUT", "INSENSITIVE", "INSERT",
    "INT", "INT1", "INT2", "INT3", "INT4", "INT8", "INTEGER", "INTERSECT",
    "INTERVAL", "INTO", "IO_AFTER_GTIDS", "IO_BEFORE_GTIDS", "IS", "ITERATE",
    "JOIN", "JSON_TABLE", "KEY", "KEYS", "KILL", "LAG", "LAST_VALUE", "LATERAL",
    "LEAD", "LEADING", "LEAVE", "LEFT", "LIKE", "LIMIT", "LINEAR", "LINES",
    "LOAD", "LOCALTIME", "LOCALTIMESTAMP", "LOCK", "LONG", "LONGBLOB",
    "LONGTEXT", "LOOP", "LOW_PRIORITY", "MATCH", "MAXVALUE", "MEDIUMBLOB",
    "MEDIUMINT", "MEDIUMTEXT", "MIDDLEINT", "MINUTE_MICROSECOND",
    "MINUTE_SECOND", "MOD", "MODIFIES", "NATURAL", "NOT", "NO_WRITE_TO_BINLOG",
    "NTH_VALUE", "NTILE", "NULL", "NUMERIC", "OF", "ON", "OPTIMIZE",
    "OPTIMIZER_COSTS", "OPTION", "OPTIONALLY", "OR", "ORDER", "OUT", "OUTER",
    "OUTFILE", "OVER", "PARTITION", "PERCENT_RANK", "PRECISION", "PRIMARY",
    "PROCEDURE", "PURGE", "RANGE", "RANK", "READ", "READS", "READ_WRITE", "REAL",
    "RECURSIVE", "REFERENCES", "REGEXP", "RELEASE", "RENAME", "REPEAT",
    "REPLACE", "REQUIRE", "RESIGNAL", "RESTRICT", "RETURN", "REVOKE", "RIGHT",
    "RLIKE", "ROW", "ROWS", "ROW_NUMBER", "SCHEMA", "SCHEMAS",
    "SECOND_MICROSECOND", "SELECT", "SENSITIVE", "SEPARATOR", "SET", "SHOW",
    "SIGNAL", "SMALLINT", "SPATIAL", "SPECIFIC", "SQL", "SQLEXCEPTION",
    "SQLSTATE", "SQLWARNING", "SQL_BIG_RESULT", "SQL_CALC_FOUND_ROWS",
    "SQL_SMALL_RESULT", "SSL", "STARTING", "STORED", "STRAIGHT_JOIN", "SYSTEM",
    "TABLE", "TERMINATED", "THEN", "TINYBLOB", "TINYINT", "TINYTEXT", "TO",
    "TRAILING", "TRIGGER", "TRUE", "UNDO", "UNION", "UNIQUE", "UNLOCK",
    "UNSIGNED", "UPDATE", "USAGE", "USE", "USING", "UTC_DATE", "UTC_TIME",
    "UTC_TIMESTAMP", "VALUES", "VARBINARY", "VARCHAR", "VARCHARACTER", "VARYING",
    "VIRTUAL", "WHEN", "WHERE", "WHILE", "WINDOW", "WITH", "WRITE", "XOR",
    "YEAR_MONTH", "ZEROFILL",
}

# https://dev.mysql.com/doc/refman/8.4/en/built-in-function-reference.html
mysql_functions = {
    "ABS", "ACOS", "ADDDATE", "ADDTIME", "AES_DECRYPT", "AES_ENCRYPT", "AND",
    "ANY_VALUE", "ASCII", "ASIN", "asynchronous_connection_failover_add_managed",
    "asynchronous_connection_failover_add_source",
    "asynchronous_connection_failover_delete_managed",
    "asynchronous_connection_failover_delete_source",
    "asynchronous_connection_failover_reset", "ATAN", "ATAN2", "AVG",
    "BENCHMARK", "BETWEEN", "BIN", "BIN_TO_UUID", "BINARY", "BIT_AND",
    "BIT_COUNT", "BIT_LENGTH", "BIT_OR", "BIT_XOR", "CAN_ACCESS_COLUMN",
    "CAN_ACCESS_DATABASE", "CAN_ACCESS_TABLE", "CAN_ACCESS_USER",
    "CAN_ACCESS_VIEW", "CASE", "CAST", "CEIL", "CEILING", "CHAR", "CHAR_LENGTH",
    "CHARACTER_LENGTH", "CHARSET", "COALESCE", "COERCIBILITY", "COLLATION",
    "COMPRESS", "CONCAT", "CONCAT_WS", "CONNECTION_ID", "CONV", "CONVERT",
    "CONVERT_TZ", "COS", "COT", "COUNT", "COUNT_DISTINCT", "CRC32", "CUME_DIST",
    "CURDATE", "CURRENT_DATE", "CURRENT_ROLE", "CURRENT_TIME",
    "CURRENT_TIMESTAMP", "CURRENT_USER", "CURTIME", "DATABASE", "DATE",
    "DATE_ADD", "DATE_FORMAT", "DATE_SUB", "DATEDIFF", "DAY", "DAYNAME",
    "DAYOFMONTH", "DAYOFWEEK", "DAYOFYEAR", "DEFAULT", "DEGREES", "DENSE_RANK",
    "DIV", "ELT", "EXISTS", "EXP", "EXPORT_SET", "EXTRACT", "ExtractValue",
    "FIELD", "FIND_IN_SET", "FIRST_VALUE", "FLOOR", "FORMAT", "FORMAT_BYTES",
    "FORMAT_PICO_TIME", "FOUND_ROWS", "FROM_BASE64", "FROM_DAYS",
    "FROM_UNIXTIME", "GeomCollection", "GeometryCollection", "GET_FORMAT",
    "GET_LOCK", "GREATEST", "GROUP_CONCAT",
    "group_replication_disable_member_action",
    "group_replication_enable_member_action",
    "group_replication_get_communication_protocol",
    "group_replication_get_write_concurrency",
    "group_replication_reset_member_actions",
    "group_replication_set_as_primary",
    "group_replication_set_communication_protocol",
    "group_replication_set_write_concurrency",
    "group_replication_switch_to_multi_primary_mode",
    "group_replication_switch_to_single_primary_mode", "GROUPING", "HEX",
    "HOUR", "ICU_VERSION", "IF", "IFNULL", "IN", "INET_ATON", "INET_NTOA",
    "INSERT", "INSTR", "INTERNAL_AUTO_INCREMENT", "INTERNAL_AVG_ROW_LENGTH",
    "INTERNAL_CHECK_TIME", "INTERNAL_CHECKSUM", "INTERNAL_DATA_FREE",
    "INTERNAL_DATA_LENGTH", "INTERNAL_DD_CHAR_LENGTH",
    "INTERNAL_GET_COMMENT_OR_ERROR", "INTERNAL_GET_ENABLED_ROLE_JSON",
    "INTERNAL_GET_HOSTNAME", "INTERNAL_GET_VIEW_WARNING_OR_ERROR",
    "INTERNAL_INDEX_COLUMN_CARDINALITY", "INTERNAL_INDEX_LENGTH",
    "INTERNAL_IS_ENABLED_ROLE", "INTERNAL_IS_MANDATORY_ROLE",
    "INTERNAL_KEYS_DISABLED", "INTERNAL_MAX_DATA_LENGTH", "INTERNAL_TABLE_ROWS",
    "INTERNAL_UPDATE_TIME", "INTERVAL", "IS", "IS_FREE_LOCK", "IS_NOT",
    "IS_NOT_NULL", "IS_NULL", "IS_USED_LOCK", "IS_UUID", "ISNULL", "JSON_ARRAY",
    "JSON_ARRAY_APPEND", "JSON_ARRAY_INSERT", "JSON_ARRAYAGG", "JSON_CONTAINS",
    "JSON_CONTAINS_PATH", "JSON_DEPTH", "JSON_EXTRACT", "JSON_INSERT",
    "JSON_KEYS", "JSON_LENGTH", "JSON_MERGE", "JSON_MERGE_PATCH",
    "JSON_MERGE_PRESERVE", "JSON_OBJECT", "JSON_OBJECTAGG", "JSON_OVERLAPS",
    "JSON_PRETTY", "JSON_QUOTE", "JSON_REMOVE", "JSON_REPLACE",
    "JSON_SCHEMA_VALID", "JSON_SCHEMA_VALIDATION_REPORT", "JSON_SEARCH",
    "JSON_SET", "JSON_STORAGE_FREE", "JSON_STORAGE_SIZE", "JSON_TABLE",
    "JSON_TYPE", "JSON_UNQUOTE", "JSON_VALID", "JSON_VALUE", "LAG", "LAST_DAY",
    "LAST_INSERT_ID", "LAST_VALUE", "LCASE", "LEAD", "LEAST", "LEFT", "LENGTH",
    "LIKE", "LineString", "LN", "LOAD_FILE", "LOCALTIME", "LOCALTIMESTAMP",
    "LOCATE", "LOG", "LOG10", "LOG2", "LOWER", "LPAD", "LTRIM", "MAKE_SET",
    "MAKEDATE", "MAKETIME", "MASTER_POS_WAIT", "MATCH", "MAX", "MBRContains",
    "MBRCoveredBy", "MBRCovers", "MBRDisjoint", "MBREquals", "MBRIntersects",
    "MBROverlaps", "MBRTouches", "MBRWithin", "MD5", "MEMBER OF", "MICROSECOND",
    "MID", "MIN", "MINUTE", "MOD", "MONTH", "MONTHNAME", "MultiLineString",
    "MultiPoint", "MultiPolygon", "NAME_CONST", "NOT", "!", "NOT BETWEEN",
    "NOT EXISTS", "NOT IN", "NOT LIKE", "NOT REGEXP", "NOW", "NTH_VALUE",
    "NTILE", "NULLIF", "OCT", "OCTET_LENGTH", "OR", "ORD", "PERCENT_RANK",
    "PERIOD_ADD", "PERIOD_DIFF", "PI", "Point", "Polygon", "POSITION", "POW",
    "POWER", "PS_CURRENT_THREAD_ID", "PS_THREAD_ID", "QUARTER", "QUOTE",
    "RADIANS", "RAND", "RANDOM_BYTES", "RANK", "REGEXP", "REGEXP_INSTR",
    "REGEXP_LIKE", "REGEXP_REPLACE", "REGEXP_SUBSTR", "RELEASE_ALL_LOCKS",
    "RELEASE_LOCK", "REPEAT", "REPLACE", "REVERSE", "RIGHT", "RLIKE",
    "ROLES_GRAPHML", "ROUND", "ROW_COUNT", "ROW_NUMBER", "RPAD", "RTRIM",
    "SCHEMA", "SEC_TO_TIME", "SECOND", "SESSION_USER", "SHA1", "SHA", "SHA2",
    "SIGN", "SIN", "SLEEP", "SOUNDEX", "SOUNDS LIKE", "SOURCE_POS_WAIT", "SPACE",
    "SQRT", "ST_Area", "ST_AsBinary", "ST_AsGeoJSON", "ST_AsText", "ST_Buffer",
    "ST_Buffer_Strategy", "ST_Centroid", "ST_Collect", "ST_Contains",
    "ST_ConvexHull", "ST_Crosses", "ST_Difference", "ST_Dimension",
    "ST_Disjoint", "ST_Distance", "ST_Distance_Sphere", "ST_EndPoint",
    "ST_Envelope", "ST_Equals", "ST_ExteriorRing", "ST_FrechetDistance",
    "ST_GeoHash", "ST_GeomCollFromText", "ST_GeomCollFromWKB", "ST_GeometryN",
    "ST_GeometryType", "ST_GeomFromGeoJSON", "ST_GeomFromText", "ST_GeomFromWKB",
    "ST_HausdorffDistance", "ST_InteriorRingN", "ST_Intersection",
    "ST_Intersects", "ST_IsClosed", "ST_IsEmpty", "ST_IsSimple", "ST_IsValid",
    "ST_LatFromGeoHash", "ST_Latitude", "ST_Length", "ST_LineFromText",
    "ST_LineFromWKB", "ST_LineInterpolatePoint", "ST_LineInterpolatePoints",
    "ST_LongFromGeoHash", "ST_Longitude", "ST_MakeEnvelope", "ST_MLineFromText",
    "ST_MLineFromWKB", "ST_MPointFromText", "ST_MPointFromWKB", "ST_MPolyFromText",
    "ST_MPolyFromWKB", "ST_NumGeometries", "ST_NumInteriorRings", "ST_NumPoints",
    "ST_Overlaps", "ST_PointAtDistance", "ST_PointFromGeoHash", "ST_PointFromText",
    "ST_PointFromWKB", "ST_PointN", "ST_PolyFromText", "ST_PolyFromWKB",
    "ST_Simplify", "ST_SRID", "ST_StartPoint", "ST_SwapXY", "ST_SymDifference",
    "ST_Touches", "ST_Transform", "ST_Union", "ST_Validate", "ST_Within", "ST_X",
    "ST_Y", "STATEMENT_DIGEST", "STATEMENT_DIGEST_TEXT", "STD", "STDDEV",
    "STDDEV_POP", "STDDEV_SAMP", "STR_TO_DATE", "STRCMP", "SUBDATE", "SUBSTR",
    "SUBSTRING", "SUBSTRING_INDEX", "SUBTIME", "SUM", "SYSDATE", "SYSTEM_USER",
    "TAN", "TIME", "TIME_FORMAT", "TIME_TO_SEC", "TIMEDIFF", "TIMESTAMP",
    "TIMESTAMPADD", "TIMESTAMPDIFF", "TO_BASE64", "TO_DAYS", "TO_SECONDS", "TRIM",
    "TRUNCATE", "UCASE", "UNCOMPRESS", "UNCOMPRESSED_LENGTH", "UNHEX",
    "UNIX_TIMESTAMP", "UpdateXML", "UPPER", "USER", "UTC_DATE", "UTC_TIME",
    "UTC_TIMESTAMP", "UUID", "UUID_SHORT", "UUID_TO_BIN",
    "VALIDATE_PASSWORD_STRENGTH", "VALUES", "VAR_POP", "VAR_SAMP", "VARIANCE",
    "VERSION", "WAIT_FOR_EXECUTED_GTID_SET", "WEEK", "WEEKDAY", "WEEKOFYEAR",
    "WEIGHT_STRING", "XOR", "YEAR", "YEARWEEK",
}


def build_protocol_stopwords() -> list[str]:
    """SQL protocol vocabulary as a CountVectorizer stop-word list.

    CountVectorizer lowercases and tokenizes before matching stop words, so we
    lowercase every keyword/function. Multi-word ("not in") and symbol ("!")
    entries can never match a single token and are dropped to avoid sklearn's
    inconsistent-stop-words warning.
    """
    words = {w.lower() for w in (mysql_keywords | mysql_functions)}
    return sorted(w for w in words if w.isalnum() or "_" in w)


def top_words_for_domain(
    queries: list[str], stopwords: list[str], top_n: int
) -> list[tuple[str, int]]:
    """Top-n most frequent non-protocol words in a domain's query pool.

    token_pattern requires tokens to start with a letter so pure numeric
    literals (ids, years, ...) don't drown out the domain vocabulary.
    """
    vectorizer = CountVectorizer(
        stop_words=stopwords,
        token_pattern=r"(?u)\b[a-zA-Z]\w+\b",
    )
    X = vectorizer.fit_transform(queries)
    counts = X.sum(axis=0).A1  # per-term total occurrences
    terms = vectorizer.get_feature_names_out()
    order = counts.argsort()[::-1][:top_n]
    return [(terms[i], int(counts[i])) for i in order]


def plot(top_words: dict[str, list[tuple[str, int]]], output_dir: Path) -> None:
    """2x2 horizontal bar panels (one per domain), top words ranked by count."""
    domains = sorted(top_words)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Domain {d}" for d in domains],
        horizontal_spacing=0.18,
        vertical_spacing=0.10,
    )

    for idx, d in enumerate(domains):
        row, col = idx // 2 + 1, idx % 2 + 1
        words, counts = zip(*top_words[d])
        # Reverse so the most frequent word sits at the top of the panel.
        fig.add_trace(
            go.Bar(
                x=counts[::-1],
                y=words[::-1],
                orientation="h",
                marker=dict(color=DOMAIN_COLORS[d]),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        width=1100,
        height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_text="Top domain words per domain (SQL protocol vocabulary removed)",
    )
    for axis in fig.layout:
        if axis.startswith("xaxis"):
            fig.layout[axis].update(showgrid=True, gridcolor="#eeeeee")
        if axis.startswith("yaxis"):
            fig.layout[axis].update(automargin=True)

    out = output_dir / "lexical_field.pdf"
    fig.write_image(str(out))
    print(f"Figure saved to {out}")


# --- pure-pgfplots figure ---------------------------------------------------
# A LaTeX twin of plot(): a 2x2 panel of horizontal bar charts (one subfigure
# per domain). Rendering the bars with pgfplots instead of plotly keeps the
# document's font for every label, tick and title.

# Full application names behind each domain letter (see the generation section).
DOMAIN_NAMES = {
    "A": "OurAirports",
    "B": "Sakila",
    "C": "AdventureWorks",
    "D": "Oracle HR",
}

TEX_FIGURE_CAPTION = (
    r"Top domain words per domain, ranked by frequency in each domain's in-domain benign "
    r"train queries, after removing the shared SQL protocol vocabulary (MySQL keywords and "
    r"built-in functions). Bar length is the token count; colours match the per-domain palette "
    r"used throughout the paper. SQL keywords are in \textit{italic}."
)

# Shared axis style for the four per-domain bar charts. Counts span tens of
# thousands, so we scale the x ticks to thousands (the "\,\cdot10^3" multiplier
# pgfplots prints) to keep the tick labels short.
LEXBAR_STYLE = (
    "  \\pgfplotsset{\n"
    "    lexbar/.style={\n"
    "      xbar,\n"
    "      width=\\linewidth, height=6cm,\n"
    "      xmin=0,\n"
    "      enlarge x limits={value=0.12,upper},\n"
    "      enlarge y limits={abs=0.6},\n"
    "      bar width=3pt,\n"
    "      ytick=data,\n"
    "      tick align=outside,\n"
    "      tick label style={font=\\tiny},\n"
    "      scaled x ticks=false,\n"
    "      x tick label style={font=\\tiny, /pgf/number format/.cd, fixed, 1000 sep={\\,}},\n"
    "      xmajorgrids, grid style={dotted},\n"
    "      title style={font=\\scriptsize},\n"
    "    },\n"
    "  }"
)

_PROTOCOL_WORDS = {w.upper() for w in (mysql_keywords | mysql_functions)}


def _tex_escape(word: str) -> str:
    """Escape the underscore, and italicise the word if it is a SQL protocol token."""
    escaped = word.replace("_", r"\_")
    if word.upper() in _PROTOCOL_WORDS:
        return f"\\textit{{{escaped}}}"
    return escaped


def _domain_subfigure(domain: str, words: list[tuple[str, int]], color_hex: str) -> str:
    """One per-domain horizontal bar chart as a ``subfigure`` (most frequent word on top).

    The word labels are drawn *inside* the panel at the left end of each bar rather than
    as y-tick labels, so long tokens grow into the plot area instead of overflowing it.
    """
    # y coords run bottom-to-top, so reverse the rank order to put rank 1 at the top.
    btt = list(reversed(words))
    coords = " ".join(f"({count},{i})" for i, (_, count) in enumerate(btt))
    cname = f"lexdom{domain}"
    # A left-anchored label at x=0 for each bar.
    labels = "\n".join(
        f"        \\node[anchor=west, font=\\tiny, inner sep=1pt] "
        f"at (axis cs:0,{i}) {{{_tex_escape(w)}}};"
        for i, (w, _) in enumerate(btt)
    )
    return (
        "  \\begin{subfigure}{0.5\\linewidth}\n"
        "    \\centering\n"
        f"    \\definecolor{{{cname}}}{{HTML}}{{{color_hex}}}\n"
        "    \\begin{tikzpicture}\n"
        "      \\begin{axis}[lexbar, ytick=\\empty]\n"
        f"        \\addplot[draw={cname}, fill={cname}, fill opacity=0.85] coordinates {{{coords}}};\n"
        f"{labels}\n"
        "      \\end{axis}\n"
        "    \\end{tikzpicture}\n"
        f"    \\caption{{{DOMAIN_NAMES.get(domain, domain)}}}\n"
        "  \\end{subfigure}"
    )


def render_tex_figure(top_words: dict[str, list[tuple[str, int]]]) -> str:
    """Render the 2x2 per-domain bar panel as a standalone pgfplots figure."""
    domains = sorted(top_words)
    subs = [
        _domain_subfigure(d, top_words[d], DOMAIN_COLORS[d].lstrip("#")) for d in domains
    ]
    # Two subfigures per row; the trailing "%" kills the inter-column space so the
    # two 0.5\linewidth panels sit flush. "\\[1ex]" breaks to the next row of the grid.
    rows = ["%\n".join(subs[i : i + 2]) for i in range(0, len(subs), 2)]
    body = "\n  \\\\[1ex]\n".join(rows)
    return (
        "\\begin{figure}[!htb]\n"
        "  \\centering\n"
        f"{LEXBAR_STYLE}\n"
        f"{body}\n"
        f"  \\caption{{{TEX_FIGURE_CAPTION}}}\n"
        "  \\label{fig:superviz26-lexical-field}\n"
        "\\end{figure}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Optional cap on benign train queries sampled per domain "
        "(default: use the full pool).",
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=None,
        help="If set, also write the 2x2 panel as a pure pgfplots/TikZ figure to this .tex path.",
    )
    args = parser.parse_args()

    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # TEMP: include SQL keywords/functions (no protocol stop-word stripping).
    stopwords = []
    print(f"Ignoring {len(stopwords)} SQL protocol tokens.")

    top_words: dict[str, list[tuple[str, int]]] = {}
    rows = []
    for domain, filename in DOMAIN_FILES.items():
        print(f"[{domain}] Loading {filename} ...")
        if args.n_samples:
            df = load_train_queries(filename, args.n_samples)
        else:
            df = pd.read_csv(DATASETS_DIR / filename, dtype=DTYPES)
            df = df[(df["split"] == "train") & (df["label"] == 0)]

        words = top_words_for_domain(
            df["full_query"].tolist(), stopwords, args.top_n
        )
        top_words[domain] = words
        for rank, (word, count) in enumerate(words, start=1):
            rows.append(
                {"domain": domain, "rank": rank, "word": word, "count": count}
            )

    csv_path = output_dir / "lexical_field_top_words.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Top words written to {csv_path}")

    plot(top_words, output_dir)

    if args.tex_output:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(render_tex_figure(top_words))
        print(f"Tex figure written to {args.tex_output}")


if __name__ == "__main__":
    main()
