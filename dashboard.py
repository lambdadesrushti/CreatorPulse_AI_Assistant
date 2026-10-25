import glob
import html
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score

BASE = os.environ.get("CP_BASE", "/content/drive/MyDrive/CreatorPulse")
GOLD, DASH = f"{BASE}/01_Data/gold_final", f"{BASE}/04_Dashboard"
DAYS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
FULL = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
LEN_ORDER = ["up to 30", "31-50", "51-70", "71-100", "100+"]
TAG_ORDER = ["none", "1-5", "6-15", "16-30", "30+"]
PRIMARY, ACCENT, ALL = "#4F46E5", "#F97316", "All niches"
STOP = list(ENGLISH_STOP_WORDS) + ["video", "videos", "like", "just", "really", "don", "ve", "ll", "amp",
                                   "https", "com", "www", "channel"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [PRIMARY, ACCENT, "#10B981", "#EC4899"]
st.set_page_config(page_title="CreatorPulse", page_icon="📈", layout="wide")

st.markdown("""<style>
.banner{background:linear-gradient(120deg,#4F46E5,#7C3AED 60%,#F97316);padding:26px 32px;border-radius:16px;color:#fff;margin-bottom:14px}
.banner h1{margin:0;font-size:2.1rem;color:#fff}.banner p{margin:6px 0 0;font-size:1.05rem;opacity:.95}
.card{border:1px solid #E5E7EB;border-radius:14px;padding:14px 18px;background:#fff;color:#111}
.card .big{font-size:1.5rem;font-weight:700;color:#4F46E5}.card .lbl{color:#6B7280;font-size:.85rem}
.tip,.do,.dont{color:#111;padding:12px 16px;border-radius:8px;margin:8px 0}
.tip{border-left:5px solid #F97316;background:#FFF7ED}
.do{border-left:5px solid #10B981;background:#ECFDF5}.dont{border-left:5px solid #EF4444;background:#FEF2F2}
.step{display:flex;gap:14px;align-items:flex-start;border:1px solid #E5E7EB;border-radius:14px;padding:14px 18px;margin:10px 0;background:#fff;color:#111}
.num{background:#4F46E5;color:#fff;border-radius:50%;min-width:34px;height:34px;display:flex;align-items:center;justify-content:center;font-weight:700}
.chip{display:inline-block;background:#EEF2FF;color:#4338CA;padding:4px 12px;border-radius:999px;margin:3px;font-size:.9rem}
</style>""", unsafe_allow_html=True)


def card(label, big, sub=""):
    return f'<div class="card"><div class="lbl">{label}</div><div class="big">{big}</div><div class="lbl">{sub}</div></div>'


def box(kind, text):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def step(n, icon, head, body):
    st.markdown(f'<div class="step"><div class="num">{n}</div><div><b>{icon} {head}</b><br>{body}</div></div>',
                unsafe_allow_html=True)


def latest_dataset():
    files = glob.glob(f"{GOLD}/final_dataset_v*.csv")
    v = lambda p: int(re.search(r"_v(\d+)\.csv$", p).group(1)) if re.search(r"_v(\d+)\.csv$", p) else -1
    return max(files, key=v)


@st.cache_data
def load_main(path):
    d = pd.read_csv(path)
    d["publish_dt"] = pd.to_datetime(d["publish_time_ist"], errors="coerce")
    d["is_high"] = d["performance_label"].astype(str).str.lower().eq("high")
    d["title_len_bucket"] = pd.cut(d["title_length"], [-1, 30, 50, 70, 100, 10000], labels=LEN_ORDER).astype(str)
    d["tag_bucket"] = pd.cut(d["tag_count"], [-1, 0, 5, 15, 30, 10000], labels=TAG_ORDER).astype(str)
    return d


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def top_words(label, n=10):
    sr = load_csv(f"{DASH}/sentiment_results.csv")
    txt = sr.loc[sr["pred_label"] == label, "comment"].dropna().astype(str)
    cv = CountVectorizer(stop_words=STOP, min_df=5, max_features=3000, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    X = cv.fit_transform(txt)
    c = pd.Series(X.sum(axis=0).A1, index=cv.get_feature_names_out())
    return c.sort_values(ascending=False).head(n).sort_values().rename_axis("word").reset_index(name="count")


@st.cache_data
def niche_trends(path, niche, min_recent):
    d = load_main(path)
    if niche != ALL:
        d = d[d["category_name"] == niche]
    tt = load_csv(f"{DASH}/trending_topics.csv")
    bad = set(tt.loc[(tt["is_generic"] == 1) | (tt["is_english_script"] == 0), "tag"])
    end = d["publish_dt"].max()
    cut1, cut2 = end - pd.Timedelta(days=90), end - pd.Timedelta(days=180)

    def counts(x):
        s = x["tags"].dropna().str.lower().str.split("|").explode().str.strip()
        return s[(s.str.len() > 2) & ~s.isin(bad)].value_counts()

    rec = counts(d[d["publish_dt"] > cut1])
    prev = counts(d[(d["publish_dt"] <= cut1) & (d["publish_dt"] > cut2)])
    t = pd.DataFrame({"recent": rec}).join(pd.DataFrame({"previous": prev})).fillna(0).astype(int)
    t = t[t["recent"] >= min_recent]
    t["change"] = t["recent"] - t["previous"]
    return t.sort_values("change", ascending=False).head(15).rename_axis("tag").reset_index()


def summary_by(frame, col, min_videos=20):
    g = frame.groupby(col).agg(videos=("video_id", "count"), median_views=("views", "median"),
                               median_engagement=("engagement_rate", "median"),
                               pct_high=("is_high", "mean")).reset_index()
    g["pct_high"] = (g["pct_high"] * 100).round(1)
    return g[g["videos"] >= min_videos]


def slots_for(frame, min_n):
    h = frame.groupby(["publish_weekday", "publish_hour"]).agg(
        n=("video_id", "count"), med=("engagement_rate", "median")).reset_index()
    return h[h["n"] >= min_n].sort_values("med", ascending=False)


def factor(frame, col, val, weight):
    g = summary_by(frame, col)
    if g.empty or val not in set(g[col]):
        return weight / 2, None, None
    best, mine = g["pct_high"].max(), float(g.loc[g[col] == val, "pct_high"].iloc[0])
    return (weight * mine / best if best > 0 else weight / 2), mine, best


def playbook(f):
    out = []
    for col, label, unit in [("title_len_bucket", "Title length", "characters"), ("tag_bucket", "Number of tags", "tags")]:
        g = summary_by(f, col)
        if len(g) >= 2:
            g = g.sort_values("pct_high", ascending=False)
            b, w = g.iloc[0], g.iloc[-1]
            out.append(("do", label, f"Aim for {b[col]} {unit}",
                        f"{b['pct_high']:.0f}% of these videos are high performers vs {w['pct_high']:.0f}% for {w[col]} {unit}."))
    for col, what in [("has_number_in_title", "a number in the title"), ("has_question_in_title", "a question in the title")]:
        g = f.groupby(col)["is_high"].agg(["mean", "count"])
        if set(g.index) == {0, 1} and g["count"].min() >= 20:
            d = (g.loc[1, "mean"] - g.loc[0, "mean"]) * 100
            kind = "do" if d >= 2 else "dont" if d <= -2 else "tip"
            head = f"Use {what}" if d >= 2 else f"Skip {what}" if d <= -2 else f"{what.capitalize()} makes little difference"
            out.append((kind, col, head, f"High performers: {g.loc[1, 'mean'] * 100:.0f}% with vs {g.loc[0, 'mean'] * 100:.0f}% without."))
    return out


PATH = latest_dataset()
df = load_main(PATH)

# ---------- sidebar ----------
st.sidebar.markdown("### Your creator profile")
niche = st.sidebar.selectbox("Your niche", [ALL] + sorted(df["category_name"].dropna().unique()))
is_all = niche == ALL
f = df if is_all else df[df["category_name"] == niche]
where = "your niche" if is_all else niche
st.sidebar.caption(f"Advice is based on {len(f):,} past trending videos in this niche.")

st.markdown(f'<div class="banner"><h1>CreatorPulse</h1><p>Your content performance and idea assistant. '
            f'Plan your next <b>{"YouTube" if is_all else html.escape(where)}</b> video with titles, timing and topics that have worked before.</p></div>',
            unsafe_allow_html=True)
t1, t2, t3, t4, t5, t6 = st.tabs(["Your Brief", "Plan a Post", "Best Time", "Topics to Ride", "Audience", "Track Progress"])
recs = playbook(f)
slots = slots_for(f, 5 if not is_all else 20)
trend = niche_trends(PATH, niche, 30 if is_all else 3)

# ---------- 1. Your Brief ----------
with t1:
    st.header("Your plan for the next video" if is_all else f"Your plan for the next {where} video")
    st.caption("Five moves, built from what worked for videos in your niche.")
    n = 1
    for kind, key, head, body in recs[:2]:
        step(n, "📝" if n == 1 else "🏷️", head, body)
        n += 1
    if not slots.empty:
        r = slots.iloc[0]
        step(n, "🕒", f"Post on {FULL[int(r['publish_weekday'])]} around {int(r['publish_hour']):02d}:00 IST",
             "Best median engagement among time slots with enough videos.")
        n += 1
    if not trend.empty:
        t = trend.iloc[0]
        step(n, "🔥", f"Work in the topic '{t['tag']}'",
             f"Used {int(t['recent'])} times in the last 90 days of the data vs {int(t['previous'])} before.")
        n += 1
    step(n, "💬", "Check your title before you publish", "Open the <b>Plan a Post</b> tab and get a score with fixes.")
    c = st.columns(3)
    c[0].markdown(card("Typical views in your niche", f"{f['views'].median():,.0f}", "per trending video"), unsafe_allow_html=True)
    c[1].markdown(card("Typical engagement", f"{f['engagement_rate'].median() * 100:.1f}%", "likes + comments per view"), unsafe_allow_html=True)
    c[2].markdown(card("Videos behind this advice", f"{len(f):,}", "from India's trending list"), unsafe_allow_html=True)
    box("tip", f"<b>What this tells us:</b> the plan combines the title, tag and timing patterns most linked with high engagement in {where}. They are associations in past trending videos, not guarantees.")

# ---------- 2. Plan a Post ----------
with t2:
    st.header("Plan a Post")
    a, b = st.columns([3, 2])
    title = a.text_input("Your title idea")
    tags_in = a.text_input("Tags (comma separated, optional)")
    day = b.selectbox("Planned day", list(FULL.values()))
    hour = b.slider("Planned hour (IST)", 0, 23, 18)
    if not title.strip():
        st.info("Type a title idea to get your scorecard.")
    else:
        nlen = len(title)
        ntag = len([x for x in tags_in.split(",") if x.strip()])
        wd = list(FULL.values()).index(day)
        pts, lines = 0, []
        p, m, bst = factor(f, "title_len_bucket", str(pd.cut([nlen], [-1, 30, 50, 70, 100, 10000], labels=LEN_ORDER)[0]), 25)
        pts += p
        lines.append(("do" if m and m >= 0.9 * bst else "dont", f"Title is {nlen} characters." +
                      (f" Titles this long are high performers {m:.0f}% of the time vs {bst:.0f}% for the best length." if m else "")))
        p, m, bst = factor(f, "tag_bucket", str(pd.cut([ntag], [-1, 0, 5, 15, 30, 10000], labels=TAG_ORDER)[0]), 20)
        pts += p
        lines.append(("do" if m and m >= 0.9 * bst else "dont", f"You entered {ntag} tags." +
                      (f" That range gives {m:.0f}% high performers vs {bst:.0f}% for the best range." if m else "")))
        for col, val, w, nm in [("has_number_in_title", int(any(ch.isdigit() for ch in title)), 15, "number"),
                                ("has_question_in_title", int("?" in title), 15, "question mark")]:
            p, m, bst = factor(f, col, val, w)
            pts += p
            lines.append(("do" if m and m >= 0.9 * bst else "dont",
                          f"Your title {'has' if val else 'has no'} {nm}." +
                          (f" {m:.0f}% high performers for this choice vs {bst:.0f}% for the better one." if m else "")))
        row = slots[(slots["publish_weekday"] == wd) & (slots["publish_hour"] == hour)]
        if not row.empty and not slots.empty:
            pts += 25 * row.iloc[0]["med"] / slots.iloc[0]["med"]
            lines.append(("do" if row.iloc[0]["med"] >= 0.9 * slots.iloc[0]["med"] else "dont",
                          f"{day} {hour:02d}:00 is {row.iloc[0]['med'] / slots.iloc[0]['med'] * 100:.0f}% as strong as the best slot."))
        else:
            pts += 12.5
            lines.append(("tip", f"Not enough videos at {day} {hour:02d}:00 to judge this slot."))
        g1, g2 = st.columns([1, 2])
        fig = go.Figure(go.Indicator(mode="gauge+number", value=round(pts), gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": PRIMARY},
            "steps": [{"range": [0, 40], "color": "#FEE2E2"}, {"range": [40, 70], "color": "#FEF3C7"},
                      {"range": [70, 100], "color": "#D1FAE5"}]}))
        fig.update_layout(height=250, margin=dict(t=20, b=0, l=20, r=20))
        g1.plotly_chart(fig)
        verdict = "Strong plan" if pts >= 70 else "Decent, tweak a few things" if pts >= 40 else "Worth reworking"
        g2.subheader(verdict)
        g2.caption("A rule-based score from past patterns in your niche. It is not a views forecast.")
        for kind, text in lines:
            box(kind, ("✅ " if kind == "do" else "⚠️ ") + text)
    st.subheader(f"What works in {where}")
    for kind, key, head, body in recs:
        box(kind, f"<b>{head}.</b> {body}")
    st.caption("These are associations in past data, not guarantees.")
    st.subheader("What the prediction model pays attention to")
    try:
        imp = load_csv(f"{GOLD}/feature_importance.csv").head(8).sort_values("importance")
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", height=320,
                               labels={"importance": "Importance in Sanobar's model", "feature": ""}))
        st.caption(f"Biggest driver: {imp.iloc[-1]['feature']}. Importance shows what the model uses, not what causes success.")
    except FileNotFoundError:
        st.info("feature_importance.csv not found.")

# ---------- 3. Best Time ----------
with t3:
    st.header("Best Time to Post (IST)")
    mn = st.slider("Only show slots with at least this many videos", 1, 100, 20 if is_all else 5, key=f"mn_{niche}")
    h = slots_for(f, mn)
    if h.empty:
        st.warning("Not enough videos. Lower the slider.")
    else:
        cols = st.columns(3)
        for i, (_, r) in enumerate(h.head(3).iterrows()):
            cols[i].markdown(card(f"Slot #{i + 1}", FULL[int(r["publish_weekday"])],
                                  f"{int(r['publish_hour']):02d}:00 IST, based on {int(r['n'])} videos"), unsafe_allow_html=True)
        pv = h.assign(day=h["publish_weekday"].map(DAYS)).pivot(index="day", columns="publish_hour", values="med").reindex(list(DAYS.values()))
        fig = px.imshow(pv, aspect="auto", color_continuous_scale="Viridis", height=320,
                        labels=dict(x="Hour (IST)", y="", color="Engagement"))
        st.plotly_chart(fig)
        box("tip", "Brighter cells did better. Blank cells have too few videos to trust. Based on trending videos, so use it as a guide.")

# ---------- 4. Topics to Ride ----------
with t4:
    st.header(f"Topics gaining ground in {where}")
    st.caption("Tags used more in the last 90 days of the data than in the 90 days before (dataset period, not today).")
    if trend.empty:
        st.info("Not enough recent videos for this niche.")
    else:
        st.markdown("".join(f'<span class="chip">#{html.escape(str(t))} +{c}</span>' for t, c in zip(trend["tag"], trend["change"])),
                    unsafe_allow_html=True)
        st.subheader("Video ideas to try")
        for _, t in trend.head(3).iterrows():
            box("do", f"Make a video built around <b>{html.escape(str(t['tag']))}</b>: used {int(t['recent'])} times recently vs {int(t['previous'])} before.")
        fig = px.bar(trend.head(10).sort_values("change"), x="change", y="tag", orientation="h", height=380,
                     labels={"change": "More uses than the previous 90 days", "tag": ""})
        st.plotly_chart(fig)

    box("tip", "<b>What this tells us:</b> these tags are used more in the latest 90 days of the data than in the 90 days before. Trends here often follow TV shows and personalities, so check that a topic fits your channel first.")

# ---------- 5. Audience ----------
with t5:
    st.header("What audiences praise and complain about")
    st.caption("Comments from a separate dataset covering many YouTube topics, so this is general audience behaviour, "
               f"not specific to {where}.")
    sk = load_csv(f"{DASH}/sentiment_by_keyword.csv")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(sk.nlargest(5, "positive").sort_values("positive"), x="positive", y="keyword", orientation="h",
                          title="Topics with the warmest comments (% positive)", color_discrete_sequence=["#10B981"], height=300))
    b.plotly_chart(px.bar(sk.nlargest(5, "negative").sort_values("negative"), x="negative", y="keyword", orientation="h",
                          title="Topics with the most critical comments (% negative)", color_discrete_sequence=["#EF4444"], height=300))
    box("tip", "<b>For creators:</b> topics that invite debate attract more criticism. Plan to moderate comments and reply early.")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(top_words("positive"), x="count", y="word", orientation="h", title="Words viewers use when happy",
                          color_discrete_sequence=["#10B981"], height=320))
    b.plotly_chart(px.bar(top_words("negative"), x="count", y="word", orientation="h", title="Words viewers use when unhappy",
                          color_discrete_sequence=["#EF4444"], height=320))
    with st.expander("How reliable is this?"):
        sr = load_csv(f"{DASH}/sentiment_results.csv")
        m = st.columns(3)
        m[0].metric("Guess-the-majority baseline", f"{sr['true_label'].value_counts(normalize=True).max():.1%}")
        m[1].metric("VADER accuracy", f"{(sr['true_label'] == sr['pred_label']).mean():.1%}")
        m[2].metric("VADER macro F1", f"{f1_score(sr['true_label'], sr['pred_label'], average='macro'):.3f}")
        st.caption("VADER can miss sarcasm and mixed-language comments.")

# ---------- 6. Track Progress ----------
with t6:
    st.header("Where do you stand?")
    st.caption(f"Compared with {where} videos that reached YouTube trending in India, so this is a high bar.")
    a, b = st.columns(2)
    mv = a.number_input("Your typical views per video", min_value=0, value=50000, step=5000)
    me = b.number_input("Your typical engagement rate (%)", min_value=0.0, value=3.0, step=0.1)
    q = f["views"].quantile([0.5, 0.75, 0.9])
    qe = f["engagement_rate"].quantile([0.5, 0.75, 0.9]) * 100
    pv, pe = (f["views"] < mv).mean() * 100, (f["engagement_rate"] * 100 < me).mean() * 100
    c = st.columns(2)
    c[0].markdown(card("Views", f"{pv:.0f}th percentile", f"Top 25% starts at {q[0.75]:,.0f} views"), unsafe_allow_html=True)
    c[1].markdown(card("Engagement", f"{pe:.0f}th percentile", f"Top 25% starts at {qe[0.75]:.1f}%"), unsafe_allow_html=True)
    box("tip", f"<b>Your next milestone:</b> " + (f"{q[0.75] / max(mv, 1):.1f}x your current views reaches the top 25% for views." if mv < q[0.75]
                                                  else "you are already in the top 25% for views.") +
        " " + (f"Engagement needs {qe[0.75] - me:+.1f} points." if me < qe[0.75] else "Engagement is already in the top 25%."))
    ladder = pd.DataFrame({"level": ["Median", "Top 25%", "Top 10%"], "views": q.values, "engagement %": qe.round(1).values})
    st.dataframe(ladder, hide_index=True)
    st.subheader("Compare with channels")
    chans = f["channel_title"].value_counts()
    chosen = st.multiselect("Channels (up to 5)", sorted(chans.index), default=chans.head(3).index.tolist(), max_selections=5)
    if chosen:
        g = f[f["channel_title"].isin(chosen)].groupby("channel_title").agg(
            videos=("video_id", "count"), median_views=("views", "median"), median_engagement=("engagement_rate", "median"),
            first=("publish_dt", "min"), last=("publish_dt", "max")).reset_index()
        g["videos_per_month"] = (g["videos"] / ((g["last"] - g["first"]).dt.days / 30.4).clip(lower=1)).round(1)
        st.dataframe(g[["channel_title", "videos", "median_views", "median_engagement", "videos_per_month"]], hide_index=True)
        st.caption("Videos per month counts only videos that reached trending.")

with st.expander("About the data"):
    st.write(f"Videos published {df['publish_dt'].min():%b %Y} to {df['publish_dt'].max():%b %Y} from India's YouTube trending list, "
             "so results lean towards successful content. Engagement = (likes + comments) / views. "
             "High performer = top 25% engagement. Findings show association, not cause. "
             "Audience data comes from a separate comment dataset.")
