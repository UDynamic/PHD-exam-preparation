import subprocess
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Config ────────────────────────────────────
st.set_page_config(page_title="Study Repo Dashboard", layout="wide", page_icon="📚")

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 6px; padding: 16px; text-align: center;
  }
  .metric-num { font-size: 2rem; font-weight: 700; color: #58a6ff; }
  .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
  h1, h2, h3 { color: #e6edf3 !important; }
  .stSelectbox label, .stMultiSelect label { color: #8b949e !important; }
  div[data-testid="stHorizontalBlock"] > div { gap: 12px; }
</style>
""", unsafe_allow_html=True)

GITHUB_COLORS = {
    "bg": "#0d1117", "surface": "#161b22", "border": "#30363d",
    "text": "#e6edf3", "muted": "#8b949e", "blue": "#58a6ff",
    "green": "#3fb950", "orange": "#d29922", "purple": "#bc8cff",
    "red": "#f85149",
}

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_commits():
    try:
        raw = subprocess.check_output(
            ["git", "log", "--pretty=format:%ad|%s", "--date=short"],
            stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        # fallback: embedded data from pasted-text.txt
        raw = """2026-05-27|CREATED interview_core_strategy.md : I'm gonna develop a core strategy for my phd interviews
2026-05-26|UPDATED interview_registration/deadlines.md: khajeh expired and Two major blockers Identified
2026-05-24|UPDATED interview_registration/deadlines.md: adding items
2026-05-24|UPDATED interview_registration/deadlines.md: added Sanjesh as the initial item.
2026-05-24|CREATED interview_registration/Deadlines.md : it's for the planning and clarity of mind
2026-05-23|UPDATED field_selection_result.md : added the resault
2026-05-23|CREATED field_selection_result.md : the result of the ranking of the universities
2026-04-11|Field selection done
2026-02-12|minimum work; revising inforcement learning
2026-02-11|23ML-NN completed
2026-02-11|working on 23ML-NN
2026-02-11|19ML-BayesClassifier completed
2026-02-11|working on 19ML-BayesClassifier
2026-02-11|got stuck on bayes error rate
2026-02-10|18ML-EnsembleLearning compeleted
2026-02-10|updated the mlflashcard prompt
2026-02-09|working on ensemble learning
2026-02-09|17ML-KNN&KDE completed
2026-02-09|working on the knn
2026-02-09|16ML-DecisionTrees completed and imported
2026-02-09|working on the Decision trees
2026-02-08|15ML-PCA completed
2026-02-08|14ML-LDA completed
2026-02-08|13ML-LinearClassifiers completed
2026-02-08|12ML-LogisticRegression completed
2026-02-08|11ML-Regularization completed
2026-02-08|10ML-holdOut&crossValidation completed
2026-02-08|09ML-Generalization completed
2026-02-08|06ML-LinearRegression completed
2026-02-08|04ML-MLE&MAP completed
2026-02-08|created ML-MLE&MAP
2026-02-08|working on the first deck, (passed DSA and ML-MLE)
2026-02-07|Tried Migrating to Latex; mass importing flashcards to Anki
2026-02-07|PCA complete
2026-02-06|working on PCA
2026-02-06|tested the master flashcard prompt on the quantum physics
2026-02-05|LDA complete
2026-02-05|reviewing LDA
2026-02-05|linear classifiers completed
2026-02-05|updated flashcard prompt
2026-02-05|dicriminative vs generative completed
2026-02-05|added the flashcards for generalization
2026-02-04|working on discriminative vs generative
2026-02-03|LR completed
2026-02-03|reviewing Logistic regression
2026-02-03|optimizing prompt for ml flashcards
2026-02-03|regularization completed
2026-02-03|reviewing regularization
2026-02-02|cross validation complete
2026-02-02|reviewing cross validation
2026-02-01|Tower of Hanoi complete
2026-02-01|solving for linear RRs
2026-02-01|adding solution for loops
2026-02-01|starting the topic of loops
2026-02-01|adding solutions for RRs
2026-01-31|adding integrals
2026-01-30|need furthure research on the integral technique
2026-01-29|adding solution for recurrence relations
2026-01-29|added geometric series
2026-01-29|adding Heights of recurrsion trees
2026-01-29|reviewing Anki
2026-01-28|Bias variance completed
2026-01-28|BGD, SGD and miniBGD completed
2026-01-28|wrote MAP in the MLE for Anki
2026-01-28|linear recursive relation complete
2026-01-28|Order of nested for loops
2026-01-27|Geometric series
2026-01-27|Solving for recurrence relations
2026-01-27|Learnt Akra-Bazzi
2026-01-27|completed ML-MLE&MAP for their first session; it's DSA day
2026-01-26|Completing ML-MAL&MAP
2026-01-25|Growth of functions done; working on recursive relations
2026-01-24|DSA intro-2 Done
2026-01-24|DSA intro Done
2026-01-24|Statistics Flashcards done
2026-01-23|Optimized and memorized ML-Evaluation, ML-LinearAlgebra, ML-MLE
2026-01-23|Ankied all the Linear Algebra
2026-01-22|AnkiBackups added to the directory
2026-01-22|setting up the old notes for Pattern recognition inside the ankiDroid
2026-01-21|need to learn about 1-nn error inequlity
2026-01-20|enough SVM for today
2026-01-20|seperated KKT conditions
2026-01-20|Planning for Svm; Subsectioning
2026-01-19|Learned Mahalonobis distance
2026-01-19|Learned Eigenvalues
2026-01-18|Learning SVM
2026-01-18|PCA done
2026-01-18|Learning on steady state of markov chains
2026-01-17|working on last test of Ensemble learning
2026-01-16|Learning theory subject done
2026-01-16|Feature selection subject done
2026-01-16|Bayesian Network subject done
2026-01-15|Markove subject done
2026-01-15|Clustering subject done
2026-01-15|Neural network is done and noted
2026-01-15|learned the backpropagation of the output and hidden layer neurons
2026-01-14|Learned the chain rule. working on sensitivity
2026-01-14|learning the backpropagation chain rule
2026-01-07|Update Pattern_recognition.md
2026-01-06|Update DSA.md
2026-01-06|Create Pattern_recognition.md
2026-01-06|Update AI.md
2025-12-31|Update AI.md
2025-12-29|Update AI.md"""

    rows = []
    for line in raw.strip().split("\n"):
        if "|" not in line:
            continue
        date, msg = line.split("|", 1)
        msg = msg.strip()

        # classify subject
        if re.search(r'\d{2}ML-|ML-|logistic|regression|regulariz|pca|lda|svm|knn|ensemble|bayes|neural|backprop|generali|classif|cluster|markov|feature select|learning theory', msg, re.I):
            subject = "ML"
        elif re.search(r'\d{2}DSA-|DSA|recurrence|recursion|hanoi|loops|growth|akra|complexity|geometric series|integral', msg, re.I):
            subject = "DSA"
        elif re.search(r'anki|flashcard|import|backup|ankidroid|deck|prompt', msg, re.I):
            subject = "Anki"
        elif re.search(r'interview|deadline|field.select|sanjesh|khajeh|phd', msg, re.I):
            subject = "Interview/Planning"
        else:
            subject = "Other"

        # classify status
        if re.search(r'complet|done|finish', msg, re.I):
            status = "completed"
        elif re.search(r'working on|reviewing|learning|studying', msg, re.I):
            status = "in_progress"
        elif re.search(r'creat|add|updat', msg, re.I):
            status = "created/updated"
        else:
            status = "other"

        # extract topic label
        topic_match = re.search(r'\d{2}(ML|DSA)-[\w&]+', msg)
        topic = topic_match.group() if topic_match else None

        rows.append({"date": pd.to_datetime(date), "message": msg,
                     "subject": subject, "status": status, "topic": topic})

    return pd.DataFrame(rows)

df = load_commits()

# ── Header ────────────────────────────────────
st.title("📚 Study Repo Dashboard")
st.markdown(f"<span style='color:{GITHUB_COLORS['muted']}'>Personal learning history · {len(df)} commits</span>", unsafe_allow_html=True)
st.divider()

# ── Top metrics ───────────────────────────────────────────────────────────────
total = len(df)
completed = len(df[df["status"] == "completed"])
days_active = df["date"].nunique()
topics_done = df[df["topic"].notna() & (df["status"] == "completed")]["topic"].nunique()

c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, total, "Total Commits"),
    (c2, completed, "Topics Completed"),
    (c3, days_active, "Active Days"),
    (c4, topics_done, "Unique Topics Done"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-num">{num}</div>
      <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Commit heatmap (GitHub green squares) ────────────────────────────────────
st.subheader("Commit Activity")

daily = df.groupby("date").size().reset_index(name="count")
daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
daily["year"] = daily["date"].dt.year
daily["dow"] = daily["date"].dt.dayofweek  # 0=Mon

# build full date range
date_range = pd.date_range(df["date"].min(), df["date"].max())
full = pd.DataFrame({"date": date_range})
full["count"] = full["date"].map(daily.set_index("date")["count"]).fillna(0).astype(int)
full["week_num"] = (full["date"] - full["date"].min()).dt.days // 7
full["dow"] = full["date"].dt.dayofweek
full["label"] = full.apply(lambda r: f"{r['date'].date()}<br>{int(r['count'])} commit{'s' if r['count']!=1 else ''}", axis=1)

heatmap = go.Figure(go.Heatmap(
    x=full["week_num"], y=full["dow"],
    z=full["count"], text=full["label"], hovertemplate="%{text}<extra></extra>",
    colorscale=[[0, "#161b22"], [0.001, "#0e4429"], [0.3, "#006d32"],
                [0.6, "#26a641"], [1.0, "#39d353"]],
    showscale=False, xgap=3, ygap=3,
))
heatmap.update_layout(
    paper_bgcolor=GITHUB_COLORS["bg"], plot_bgcolor=GITHUB_COLORS["bg"],
    height=180, margin=dict(l=40, r=10, t=10, b=10),
    yaxis=dict(tickvals=list(range(7)),ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
               color=GITHUB_COLORS["muted"], showgrid=False),
    xaxis=dict(visible=False),
    font=dict(color=GITHUB_COLORS["muted"]),
)
st.plotly_chart(heatmap, use_container_width=True)

# ── Subject breakdown + Status breakdown ─────────────────────────────────────
st.subheader("Breakdown")
col_a, col_b = st.columns(2)

with col_a:
    subj_counts = df["subject"].value_counts().reset_index()
    subj_counts.columns = ["subject", "count"]
    colors = [GITHUB_COLORS["blue"], GITHUB_COLORS["green"],
              GITHUB_COLORS["purple"], GITHUB_COLORS["orange"], GITHUB_COLORS["red"]]
    fig_subj = px.bar(subj_counts, x="subject", y="count",
                      color="subject", color_discrete_sequence=colors,
                      title="Commits by Subject")
    fig_subj.update_layout(
        paper_bgcolor=GITHUB_COLORS["bg"], plot_bgcolor=GITHUB_COLORS["bg"],
        showlegend=False, font=dict(color=GITHUB_COLORS["muted"]),
        title_font_color=GITHUB_COLORS["text"],
        xaxis=dict(color=GITHUB_COLORS["muted"], showgrid=False),
        yaxis=dict(color=GITHUB_COLORS["muted"], gridcolor=GITHUB_COLORS["border"]),
    )
    st.plotly_chart(fig_subj, use_container_width=True)

with col_b:
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig_status = px.pie(status_counts, names="status", values="count",
                        title="Commit Status Distribution",
                        color_discrete_sequence=[GITHUB_COLORS["green"], GITHUB_COLORS["blue"],
                                                 GITHUB_COLORS["orange"], GITHUB_COLORS["muted"]])
    fig_status.update_layout(
        paper_bgcolor=GITHUB_COLORS["bg"], plot_bgcolor=GITHUB_COLORS["bg"],
        font=dict(color=GITHUB_COLORS["muted"]),
        title_font_color=GITHUB_COLORS["text"],
        legend=dict(font=dict(color=GITHUB_COLORS["muted"])),
    )
    st.plotly_chart(fig_status, use_container_width=True)

# ── Topic completion timeline ─────────────────────────────────────────────────
st.subheader("Topic Completion Timeline")

completed_topics = df[df["status"] == "completed"].dropna(subset=["topic"]).copy()
completed_topics = completed_topics.sort_values("date")

if not completed_topics.empty:
    fig_timeline = go.Figure()
    subject_color = {"ML": GITHUB_COLORS["blue"], "DSA": GITHUB_COLORS["green"],
                     "Anki": GITHUB_COLORS["purple"], "Interview/Planning": GITHUB_COLORS["orange"]}
    for _, row in completed_topics.iterrows():
        color = subject_color.get(row["subject"], GITHUB_COLORS["muted"])
        fig_timeline.add_trace(go.Scatter(
            x=[row["date"]], y=[row["topic"]],
            mode="markers", marker=dict(size=12, color=color, symbol="circle"),
            name=row["subject"], showlegend=False,
            hovertemplate=f"<b>{row['topic']}</b><br>{row['date'].date()}<extra></extra>",
        ))
    fig_timeline.update_layout(
        paper_bgcolor=GITHUB_COLORS["bg"], plot_bgcolor=GITHUB_COLORS["bg"],
        height=max(300, len(completed_topics) * 22),
        font=dict(color=GITHUB_COLORS["muted"]),
        xaxis=dict(color=GITHUB_COLORS["muted"], gridcolor=GITHUB_COLORS["border"]),
        yaxis=dict(color=GITHUB_COLORS["muted"], showgrid=False),
        margin=dict(l=180, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ── Anki activity ─────────────────────────────────────────────────────────────
st.subheader("Anki / Flashcard Activity")

anki_df = df[df["subject"] == "Anki"].copy()
if not anki_df.empty:
    anki_daily = anki_df.groupby("date").size().reset_index(name="count")
    fig_anki = px.line(anki_daily, x="date", y="count",
                       markers=True, color_discrete_sequence=[GITHUB_COLORS["purple"]])
    fig_anki.update_layout(
        paper_bgcolor=GITHUB_COLORS["bg"], plot_bgcolor=GITHUB_COLORS["bg"],
        font=dict(color=GITHUB_COLORS["muted"]),
        xaxis=dict(color=GITHUB_COLORS["muted"], gridcolor=GITHUB_COLORS["border"]),
        yaxis=dict(color=GITHUB_COLORS["muted"], gridcolor=GITHUB_COLORS["border"]),
        margin=dict(l=40, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_anki, use_container_width=True)
else:
    st.info("No Anki commits found.")

# ── Interview prep timeline ───────────────────────────────────────────────────
st.subheader("Interview & Planning Milestones")

interview_df = df[df["subject"] == "Interview/Planning"].sort_values("date")
if not interview_df.empty:
    for _, row in interview_df.iterrows():
        dot_color = GITHUB_COLORS["orange"]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;padding:6px 0;border-bottom:1px solid {GITHUB_COLORS['border']}'>"
            f"<span style='color:{GITHUB_COLORS['muted']};font-size:0.8rem;min-width:90px'>{row['date'].date()}</span>"
            f"<span style='color:{dot_color}'>●</span>"
            f"<span style='color:{GITHUB_COLORS['text']};font-size:0.9rem'>{row['message']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# ── Raw log ───────────────────────────────────────────────────────────────────
with st.expander("📋 Raw Commit Log"):
    subject_filter = st.multiselect("Filter by subject", df["subject"].unique().tolist(),
                                    default=df["subject"].unique().tolist())
    filtered = df[df["subject"].isin(subject_filter)].sort_values("date", ascending=False)
    st.dataframe(
        filtered[["date", "subject", "status", "message"]].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "subject": st.column_config.TextColumn("Subject"),
            "status": st.column_config.TextColumn("Status"),
            "message": st.column_config.TextColumn("Commit Message", width="large"),
        }
    )
