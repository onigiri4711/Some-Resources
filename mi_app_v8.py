import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import numpy as np

st.set_page_config(
    page_title="MI Resource Universe",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (static string only, no dynamic building) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
html, body, [class*="css"] {
    background-color: #050a14 !important;
    color: #e0e0e0;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
}
.stApp { background-color: #050a14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem; max-width: 1400px; }
.stTabs [data-baseweb="tab-list"] {
    background: #060c18;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: rgba(255,255,255,0.4);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    padding: 12px 22px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #4d96ff !important;
    border-bottom: 2px solid #4d96ff !important;
}
.resource-card {
    background: #0a1220;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 24px;
    margin-top: 8px;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
}
.resource-card .tag { font-size: 10px; letter-spacing: 3px; margin-bottom: 8px; }
.resource-card h3 {
    font-size: 17px; font-weight: 700; color: #fff;
    margin: 0 0 6px 0; line-height: 1.35;
}
.resource-card p {
    font-size: 12px; color: rgba(180,180,200,0.65);
    line-height: 1.85; margin: 0;
}
.visit-btn {
    display: inline-block; margin-top: 16px; padding: 9px 20px;
    border-radius: 6px; font-size: 11px; letter-spacing: 2px;
    font-family: 'Share Tech Mono', monospace; text-decoration: none;
    font-weight: 700; transition: opacity 0.2s;
}
.visit-btn:hover { opacity: 0.8; }
.placeholder {
    color: rgba(255,255,255,0.18); font-size: 12px; line-height: 2.2;
    padding: 28px 8px; font-family: 'Share Tech Mono', monospace;
}
/* Further Reading native cards */
.fr-card {
    background: #0a1220;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
    font-family: 'Share Tech Mono', monospace;
}
.stButton button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: rgba(200,200,220,0.7) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1px !important;
    padding: 4px 14px !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
RESOURCES = [
    {
        "id": "framework", "label": "Transformer Circuits\nFramework",
        "year": 2021, "type": "paper", "color": "#ff6b6b",
        "subtitle": "The Foundation",
        "tagline": "The 'Bible' of the field — introduces the QK and OV circuit decomposition.",
        "title": "A Mathematical Framework for Transformer Circuits",
        "author": "Elhage et al., 2021",
        "url": "https://transformer-circuits.pub/2021/framework/",
        "desc": (
            "This paper laid down the building blocks of Mechanistic Interpretability. "
            "There was some prior work in reverse engineering vision models done by C. Olah "
            "in his Circuits Thread. This paper focuses more deeply on the mechanistic "
            "interpretability of LLMs — to be more specific, the architecture running behind "
            "an LLM, i.e. the Transformer. Here the Transformer is broken down on the basis "
            "of its attention layers. The main aim of the paper is to break the concept of a "
            "Transformer being a black box and make it interpretable in human standards of comprehensibility."
        ),
        "extra_links": [
            {"label": "Circuits Thread (C. Olah)",
             "url": "https://distill.pub/2020/circuits/zoom-in/"},
        ],
    },
    {
        "id": "mono", "label": "Towards\nMonosemanticity",
        "year": 2023, "type": "paper", "color": "#ffd93d",
        "subtitle": "The Feature Breakthrough",
        "tagline": "Explains how Sparse Autoencoders (SAEs) solve the problem of messy neurons.",
        "title": "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning",
        "author": "Bricken et al., 2023",
        "url": "https://transformer-circuits.pub/2023/monosemantic-features",
        "desc": (
            "Polysemanticity makes it really difficult to reason and account for the behaviour "
            "of each individual Neuron — it becomes more and more difficult to interpret the "
            "behaviour as to why it is learning more number of features than it has dimensions "
            "in each such neuron. What this paper does is implement a sparse autoencoder. "
            "It basically splits the features in these neurons into finer interpretable features. "
            "This approach of introducing sparsity in an autoencoder enables it to undo the "
            "superposition. You can carry out the tutorial to try it out and practice yourself "
            "to understand it better."
        ),
        "extra_links": [
            {"label": "Practice Colab Tutorial",
             "url": "https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn"},
        ],
    },
    {
        "id": "gemma2", "label": "Gemma Scope 2",
        "year": 2025, "type": "paper", "color": "#6bcb77",
        "subtitle": "The Scaling Milestone",
        "tagline": "A massive release of SAEs for models up to 27B parameters — a pre-mapped atlas of neural features.",
        "title": "Gemma Scope 2: Open Interpretability at Scale",
        "author": "Google DeepMind, 2025",
        "url": "https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf",
        "desc": (
            "This is a paper discussing the toolkit by DeepMind's Language Model Interpretability "
            "Team, released with pre-trained Sparse Autoencoders and transcoders on the Gemma 3 model. "
            "The Gemma 3 is a relatively new, modern and open source family of lightweight models "
            "with 27B parameters. Gemma Scope 2 allows you to get a transparent look at the chain "
            "of thought of a model — this will allow further researchers to engineer a way to "
            "prevent jailbreaking."
        ),
        "extra_links": [
            {"label": "DeepMind Blog Post",
             "url": "https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/"},
        ],
    },
    {
        "id": "introspect", "label": "Signs of\nIntrospection",
        "year": 2025, "type": "paper", "color": "#4d96ff",
        "subtitle": "The Current Frontier",
        "tagline": "Investigates whether models can internally monitor their own truthfulness.",
        "title": "Signs of Introspection in Large Language Models",
        "author": "Anthropic, 2025",
        "url": "https://transformer-circuits.pub/2025/introspection/index.html",
        "desc": (
            "The paper tries to answer: Is an LLM aware of its internal states? "
            "In general if you ask a model what it is thinking about, you will get some response — but "
            "how do we know if it's genuine? What if its response is just meant to sound genuine? "
            "The paper tries to crack down on whether the LLM is truly introspecting or not. "
            "Suppose within the neurons it's currently thinking about bananas — if it responds that, "
            "then it will be truly introspecting. But this output needs to come directly from that neuron "
            "it is thinking in. The response should not be influenced by later activations and layers, "
            "otherwise it cannot truly be introspecting. Hence they carried out experiments to verify "
            "that. This approach brings about a form of transparency in LLMs."
        ),
        "extra_links": [],
    },
    {
        "id": "tl", "label": "TransformerLens",
        "year": 2022, "type": "tool", "color": "#ff9f43",
        "best_for": "The industry standard for exploratory analysis and activation patching in PyTorch.",
        "title": "TransformerLens", "author": "Neel Nanda",
        "url": "https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb",
        "desc": (
            "This library was made by Neel Nanda, one of the central figureheads in the field of "
            "Mechanistic Interpretability — almost spearheading this field of research. "
            "This library allows you to put a microscope on and understand the internal workings of "
            "GPT-2 like models. To get a hands-on experience with this library you can directly "
            "implement it through the demo in the provided Colab notebook by Neel Nanda himself."
        ),
        "extra_links": [
            {"label": "GitHub / Docs",
             "url": "https://github.com/neelnanda-io/TransformerLens"},
        ],
    },
    {
        "id": "nnsight", "label": "nnsight",
        "year": 2023, "type": "tool", "color": "#ee5a24",
        "best_for": "High-performance interpretability on massive models (like Llama-3 405B) via remote execution.",
        "title": "nnsight", "author": "NDIF",
        "url": "https://nnsight.net/walkthroughs/",
        "desc": (
            "It is a library used to interpret as well as manipulate whatever happens inside the "
            "internal states of a model. It directly works with PyTorch and can wrap any model and "
            "trace through the model and access its hidden states and manipulate the output to measure "
            "how it affects the model overall. The following is a link to a set of walkthrough "
            "tutorials within Colab notebooks from the nnsight website covering all the crucial "
            "parts and concepts for Mechanistic Interpretability."
        ),
        "extra_links": [],
    },
    {
        "id": "gemscope", "label": "Gemma Scope",
        "year": 2024, "type": "tool", "color": "#a29bfe",
        "best_for": "An open-source suite of over 1 trillion SAE parameters for analysing the Gemma model family.",
        "title": "Gemma Scope", "author": "Google DeepMind",
        "url": "https://www.neuronpedia.org/gemma-scope",
        "desc": (
            "It is a toolkit used to shed light on the inner workings of a language model — "
            "here it is specifically meant for Gemma, the lightweight family of open models by DeepMind. "
            "You can refer to the interactive demo on Neuronpedia to see how it would work. "
            "If you would like to implement it through hardcoding, you can follow the Colab walkthrough."
        ),
        "extra_links": [
            {"label": "Colab Walkthrough",
             "url": "https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp"},
        ],
    },
    {
        "id": "neuro", "label": "Neuroscope",
        "year": 2023, "type": "tool", "color": "#fd79a8",
        "best_for": "A web-based tool to browse high-activating examples for every neuron in GPT-2.",
        "title": "Neuroscope", "author": "Neel Nanda",
        "url": "https://neuroscope.io/gpt2-medium/index.html",
        "desc": (
            "This is another tool made by Neel Nanda for viewing the models' highest activations "
            "of each neuron layer by layer. You can also tinker with it in code for each neuron and "
            "visualise its activations. It includes models such as GPT-2 in all its sizes "
            "(small, medium, large, XL), and some other models that were trained specifically "
            "for interpretability research."
        ),
        "extra_links": [],
    },
    {
        "id": "arena", "label": "ARENA",
        "year": 2022, "type": "community", "color": "#00cec9",
        "title": "ARENA (AI Alignment & Reconstruction)", "author": "arena.education",
        "url": "https://www.arena.education/",
        "desc": (
            "They provide in-person programming bootcamps, equipping people with the skills, "
            "community and confidence to contribute to technical AI safety. They give a rigorous, "
            "coding-heavy curriculum that covers everything from training Transformers from scratch "
            "to advanced Mech Interp techniques. There is no single profile that they look for "
            "specifically — recent iterations of ARENA have had successful applicants come from "
            "diverse academic and professional backgrounds. It's a good place to start; all resources "
            "are openly available in their curriculum on the website."
        ),
        "extra_links": [
            {"label": "Curriculum",
             "url": "https://www.arena.education/curriculum"},
        ],
    },
    {
        "id": "quickstart", "label": "Neel Nanda's\nQuickstart",
        "year": 2022, "type": "community", "color": "#fdcb6e",
        "title": "Neel Nanda's Quickstart Guide", "author": "neelnanda.io",
        "url": "https://www.neelnanda.io/mechanistic-interpretability/quickstart-old",
        "desc": (
            "It is the most popular entry point for software engineers getting into Mechanistic "
            "Interpretability. It includes the 200 Concrete Open Problems in Mechanistic "
            "Interpretability sequence — an invaluable resource for anyone looking for project "
            "ideas to work on."
        ),
        "extra_links": [
            {"label": "200 Concrete Open Problems",
             "url": "https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability"},
        ],
    },
    {
        "id": "af", "label": "Alignment Forum",
        "year": 2018, "type": "community", "color": "#55efc4",
        "title": "AI Alignment Forum", "author": "alignmentforum.org",
        "url": "https://www.alignmentforum.org/",
        "desc": (
            "It is the central hub for high-level discussion and the latest research pre-prints "
            "in the interpretability and safety space, as well as blog posts by some of the "
            "leading scientists in the field."
        ),
        "extra_links": [
            {"label": "200 Open Problems",
             "url": "https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability"},
        ],
    },
]

EXTRA_LINKS = [
    {
        "label": "transformer-circuits.pub",
        "color": "#ff6b6b",
        "url":   "https://transformer-circuits.pub/",
        "desc": (
            "This thread only became a thing due to earlier work done on Circuits in Vision Model "
            "InceptionV1, which was also done by Chris Olah (also at Anthropic). This thread "
            "consists of all the developments, starting with the Mathematical Framework in "
            "Transformer Circuits. Their latest work discusses models explaining there own activations "
            "using this model called as Activation Oracles."
            "This model allows to see the internal workings of an LLM and check"
            "if their was any misallignment  involved in the finetuning of the model."
        ),
        "extra_links": [],
    },
    {
        "label": "Toy Models of Superposition",
        "color": "#ffd93d",
        "url":   "https://transformer-circuits.pub/2022/toy_model/index.html",
        "desc": (
            "It deals with polysemanticity in toy models, which are essentially smaller in scale "
            "compared to the larger models. A great way to build intuition for superposition "
            "before tackling the full-scale SAE papers."
        ),
        "extra_links": [
            {"label": "Try the Code (Colab)",
             "url": "https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb"},
        ],
    },
    {
        "label": "Interpreting GPT: The Logit Lens",
        "color": "#a29bfe",
        "url":   "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens",
        "desc": (
            "This post discusses a technique to see what the model believes the next word will be "
            "after each intermediate step of its processing — a simple but powerful interpretability tool."
        ),
        "extra_links": [
            {"label": "Run the Code (Colab)",
             "url": "https://colab.research.google.com/drive/1-nOE-Qyia3ElM17qrdoHAtGmLCPUZijg"},
        ],
    },
]

EDGES = [
    ("framework","tl"),("framework","mono"),("framework","neuro"),
    ("mono","gemscope"),("mono","nnsight"),("gemscope","gemma2"),
    ("tl","arena"),("nnsight","arena"),("arena","af"),
    ("introspect","framework"),("introspect","mono"),
    ("gemma2","gemscope"),("framework","af"),
    ("af","quickstart"),("arena","quickstart"),
]

res_by_id  = {r["id"]: r for r in RESOURCES}
type_emoji = {"paper":"📄","tool":"🔧","community":"🌐"}
BG, GRID   = "#050a14","#0a1220"

def hex_to_rgba(hex_color, alpha=1.0):
    h = hex_color.lstrip("#")
    rv,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return "rgba(%d,%d,%d,%.2f)" % (rv,g,b,alpha)

PLOTLY_BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=GRID,
    font=dict(family="Courier New",color="#e0e0e0"),
    hoverlabel=dict(bgcolor="#0a1220",bordercolor="rgba(255,255,255,0.15)",
                    font=dict(color="#e0e0e0",size=12,family="Courier New")),
)

if "selected" not in st.session_state:
    st.session_state.selected = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#050a14 0%,#0d1a2e 60%,#050a14 100%);
            border-radius:14px;padding:36px 40px 28px;
            border:1px solid rgba(255,255,255,0.07);margin-bottom:24px;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
              letter-spacing:4px;color:#4d96ff;margin-bottom:8px;">
    MECHANISTIC INTERPRETABILITY
  </div>
  <h1 style="font-family:'Share Tech Mono',monospace;font-size:30px;font-weight:900;margin:0 0 10px;
             background:linear-gradient(90deg,#ff6b6b,#ffd93d,#6bcb77,#4d96ff,#a29bfe);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    Resources &amp; Further Reading
  </h1>
  <p style="font-family:'Share Tech Mono',monospace;color:rgba(180,180,200,0.55);
            font-size:12px;margin:0 0 18px;line-height:1.8;">
    An interactive map of the key papers, tools, and communities that define the field.
    Click any node or marker to read the full write-up and visit the resource directly.
  </p>
  <span style="font-size:11px;color:#ff6b6b;background:rgba(255,107,107,.1);
               border:1px solid rgba(255,107,107,.3);border-radius:20px;padding:4px 14px;margin-right:10px;">
    &#128196; 4 Papers</span>
  <span style="font-size:11px;color:#ff9f43;background:rgba(255,159,67,.1);
               border:1px solid rgba(255,159,67,.3);border-radius:20px;padding:4px 14px;margin-right:10px;">
    &#128296; 4 Tools</span>
  <span style="font-size:11px;color:#00cec9;background:rgba(0,206,201,.1);
               border:1px solid rgba(0,206,201,.3);border-radius:20px;padding:4px 14px;margin-right:10px;">
    &#127760; 3 Communities</span>
  <span style="font-size:11px;color:#a29bfe;background:rgba(162,155,254,.1);
               border:1px solid rgba(162,155,254,.3);border-radius:20px;padding:4px 14px;">
    &#128279; 3 Extra Links</span>
</div>
""", unsafe_allow_html=True)

# ── Side panel renderer (tabs 1-3) — uses only safe static HTML templates ─────
def render_panel(resource_id):
    if resource_id is None:
        st.markdown(
            '<div class="resource-card">'
            '<p class="placeholder">'
            '&larr; Click any node or point<br>to read the full write-up<br>and open the resource.'
            '</p></div>',
            unsafe_allow_html=True)
        return

    r   = res_by_id[resource_id]
    col = r["color"]
    t   = type_emoji[r["type"]]

    # Build each piece as a safe variable — no apostrophes inside HTML attributes
    parts = []
    parts.append('<div class="resource-card" style="border-color:' + col + '33;">')
    parts.append('<div class="tag" style="color:' + col + ';">' + t + ' ' + r["type"].upper() + '</div>')

    if r.get("subtitle"):
        parts.append(
            '<div style="display:inline-block;margin-bottom:10px;padding:3px 12px;'
            'border-radius:20px;font-size:9px;letter-spacing:2px;font-weight:700;'
            'background:' + col + '22;color:' + col + ';border:1px solid ' + col + '55;">'
            + r["subtitle"] + '</div>')

    parts.append('<h3>' + r["title"] + '</h3>')
    parts.append(
        '<div style="font-size:10px;color:rgba(255,255,255,0.3);'
        'letter-spacing:2px;margin-bottom:8px;">' + r["author"] + '</div>')

    if r.get("tagline"):
        parts.append(
            '<div style="font-size:11px;color:rgba(200,200,220,0.5);'
            'font-style:italic;margin-bottom:10px;line-height:1.5;">'
            + r["tagline"] + '</div>')

    if r.get("best_for"):
        parts.append(
            '<div style="margin:10px 0 14px;padding:10px 14px;border-radius:8px;'
            'background:rgba(255,255,255,0.04);border-left:3px solid ' + col + ';'
            'font-size:11px;color:rgba(200,200,220,0.7);line-height:1.6;">'
            '<span style="color:' + col + ';letter-spacing:2px;font-size:9px;">BEST FOR</span><br>'
            + r["best_for"] + '</div>')

    parts.append('<p>' + r["desc"] + '</p>')
    parts.append('<div style="margin-top:16px;">')
    parts.append(
        '<a href="' + r["url"] + '" target="_blank" class="visit-btn" '
        'style="background:' + col + '22;color:' + col + ';border:1px solid ' + col + '66;">'
        'Visit Resource &#8599;</a>')

    for lnk in r.get("extra_links", []):
        parts.append(
            '<a href="' + lnk["url"] + '" target="_blank" class="visit-btn" '
            'style="background:rgba(255,255,255,0.07);color:rgba(255,255,255,0.5);'
            'border:1px solid rgba(255,255,255,0.15);margin-left:8px;">'
            + lnk["label"] + ' &#8599;</a>')

    parts.append('</div></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🕸  Knowledge Graph", "⏱  Timeline",
    "📊  Radar Profile",   "🔗  Further Reading",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<p style="font-size:11px;letter-spacing:2px;color:rgba(255,255,255,0.25);'
        'margin:12px 0 4px;">CLICK ANY NODE TO READ THE WRITE-UP</p>',
        unsafe_allow_html=True)
    col_chart, col_panel = st.columns([2,1], gap="medium")

    with col_chart:
        G = nx.DiGraph()
        G.add_nodes_from([r["id"] for r in RESOURCES])
        G.add_edges_from(EDGES)
        np.random.seed(42)
        pos = nx.spring_layout(G, k=2.2, iterations=120, seed=42)

        edge_traces = []
        for src, dst in EDGES:
            x0,y0=pos[src]; x1,y1=pos[dst]
            edge_traces.append(go.Scatter(
                x=[x0,x1,None],y=[y0,y1,None],mode="lines",
                line=dict(width=1.4,color=hex_to_rgba(res_by_id[src]["color"],0.25)),
                hoverinfo="none",showlegend=False))

        node_traces = []
        for rtype in ["paper","tool","community"]:
            subset=[r for r in RESOURCES if r["type"]==rtype]
            size={"paper":30,"tool":22,"community":26}[rtype]
            node_traces.append(go.Scatter(
                x=[pos[r["id"]][0] for r in subset],
                y=[pos[r["id"]][1] for r in subset],
                mode="markers+text",
                name=type_emoji[rtype]+" "+rtype.capitalize(),
                text=[r["label"].replace("\n","<br>") for r in subset],
                textposition="top center",
                textfont=dict(color="rgba(220,220,220,0.85)",size=9,family="Courier New"),
                customdata=[r["id"] for r in subset],
                hovertemplate="<b>%{text}</b><br><i>Click to read write-up</i><extra></extra>",
                marker=dict(size=size,
                    color=[hex_to_rgba(r["color"],0.65) for r in subset],
                    line=dict(width=2,color=[r["color"] for r in subset]))))

        fig1=go.Figure(data=edge_traces+node_traces)
        fig1.update_layout(
            **PLOTLY_BASE,height=500,margin=dict(l=10,r=10,t=20,b=10),
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            legend=dict(font=dict(color="#aaa",size=10,family="Courier New"),
                bgcolor="rgba(10,18,32,0.85)",bordercolor="rgba(255,255,255,0.1)",borderwidth=1),
            clickmode="event")

        clicked1=st.plotly_chart(fig1,use_container_width=True,
                                  key="graph_chart",on_select="rerun",selection_mode="points")
        if clicked1 and clicked1.selection and clicked1.selection.get("points"):
            pt=clicked1.selection["points"][0]
            if pt.get("customdata"):
                st.session_state.selected=pt["customdata"]

    with col_panel:
        render_panel(st.session_state.selected)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<p style="font-size:11px;letter-spacing:2px;color:rgba(255,255,255,0.25);'
        'margin:12px 0 4px;">CLICK ANY MARKER TO READ THE WRITE-UP</p>',
        unsafe_allow_html=True)
    col_tl, col_tl_panel = st.columns([2, 1], gap="medium")

    with col_tl:
        sorted_res = sorted(RESOURCES, key=lambda r: (r["year"], r["id"]))
        n = len(sorted_res)
        year_idx = {}
        xs, ys, colors, labels, cdata_tl = [], [], [], [], []
        conn_x, conn_y = [], []

        for i, r in enumerate(sorted_res):
            idx = year_idx.get(r["year"], 0)
            year_idx[r["year"]] = idx + 1
            side = 1 if idx % 2 == 0 else -1
            offset = (idx // 2 + 1) * 0.85 * side
            y_pos = i * 1.2
            xs.append(offset)
            ys.append(y_pos)
            colors.append(r["color"])
            labels.append(r["label"].replace("\n", "<br>"))
            cdata_tl.append(r["id"])
            conn_x += [0, offset, None]
            conn_y += [y_pos, y_pos, None]

        yr_range = [-1, (n - 1) * 1.2 + 1]
        tl_data = [
            go.Scatter(x=[0, 0], y=yr_range, mode="lines",
                line=dict(color="rgba(255,255,255,0.15)", width=2),
                hoverinfo="none", showlegend=False),
            go.Scatter(x=conn_x, y=conn_y, mode="lines",
                line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot"),
                hoverinfo="none", showlegend=False),
        ]
        for rtype, sym in [("paper","circle"),("tool","diamond"),("community","star")]:
            mask = [i for i, r in enumerate(sorted_res) if r["type"] == rtype]
            tl_data.append(go.Scatter(
                x=[xs[i] for i in mask], y=[ys[i] for i in mask],
                mode="markers+text",
                name=type_emoji[rtype]+" "+rtype.capitalize(),
                text=[labels[i] for i in mask],
                textposition=["middle right" if xs[i] >= 0 else "middle left" for i in mask],
                textfont=dict(color="rgba(220,220,220,0.8)", size=10, family="Courier New"),
                customdata=[cdata_tl[i] for i in mask],
                hovertemplate="<b>%{text}</b><br><i>Click to read write-up</i><extra></extra>",
                marker=dict(size=14, symbol=sym,
                    color=[hex_to_rgba(colors[i], 0.7) for i in mask],
                    line=dict(width=2, color=[colors[i] for i in mask]))))

        max_offset = max(abs(x) for x in xs) if xs else 1
        fig2 = go.Figure(data=tl_data)
        fig2.update_layout(
            **PLOTLY_BASE, height=800, autosize=True,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-(max_offset + 2.2), max_offset + 2.2]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-1.5, (n - 1) * 1.2 + 1.5]),
            legend=dict(font=dict(color="#aaa", size=10, family="Courier New"),
                bgcolor="rgba(10,18,32,0.85)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            clickmode="event")

        # Year labels — one per unique year at average y of that year's items
        year_positions = {}
        for i, r in enumerate(sorted_res):
            year_positions.setdefault(r["year"], []).append(ys[i])
        for yr, positions in year_positions.items():
            avg_y = sum(positions) / len(positions)
            fig2.add_annotation(x=0, y=avg_y, text="<b>"+str(yr)+"</b>", showarrow=False,
                font=dict(color="rgba(255,255,255,0.5)", size=10, family="Courier New"),
                bgcolor="#0a1220", bordercolor="rgba(255,255,255,0.12)", borderpad=4)

        # Single chart render — outside the loop
        clicked2 = st.plotly_chart(fig2, use_container_width=True,
                                   key="tl_chart", on_select="rerun",
                                   selection_mode="points")
        if clicked2 and clicked2.selection and clicked2.selection.get("points"):
            pt = clicked2.selection["points"][0]
            if pt.get("customdata"):
                st.session_state.selected = pt["customdata"]

    with col_tl_panel:
        render_panel(st.session_state.selected)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RADAR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<p style="font-size:11px;letter-spacing:2px;color:rgba(255,255,255,0.25);'
        'margin:12px 0 4px;">CLICK ANY POLYGON TO READ THE WRITE-UP &middot; TOGGLE LEGEND TO COMPARE</p>',
        unsafe_allow_html=True)
    col_radar,col_radar_panel=st.columns([2,1],gap="medium")

    AXES=["Depth","Accessibility","Tooling","Community","Recency"]
    SCORES={
        "framework":[95,45,30,70,40],"mono":[90,55,60,75,70],
        "gemma2":[85,65,80,60,95],"introspect":[80,50,35,65,95],
        "tl":[70,85,95,90,65],"nnsight":[68,75,88,70,72],
        "gemscope":[75,78,85,65,85],"arena":[75,95,80,99,75],
        "af":[80,70,40,98,60],
    }

    with col_radar:
        fig3=go.Figure()
        for rid,scores in SCORES.items():
            r=res_by_id[rid]; label=r["label"].replace("\n"," ")
            vals=scores+[scores[0]]; cats=AXES+[AXES[0]]
            fig3.add_trace(go.Scatterpolar(
                r=vals,theta=cats,fill="toself",
                name=type_emoji[r["type"]]+" "+label,
                customdata=[rid]*len(vals),
                line=dict(color=r["color"],width=2),
                fillcolor=hex_to_rgba(r["color"],0.12),
                hovertemplate="<b>"+label+"</b><br>%{theta}: <b>%{r}</b><extra></extra>"))
        fig3.update_layout(
            **PLOTLY_BASE,height=560,
            polar=dict(bgcolor=GRID,
                radialaxis=dict(visible=True,range=[0,100],
                    tickfont=dict(color="rgba(255,255,255,0.2)",size=9,family="Courier New"),
                    gridcolor="rgba(255,255,255,0.07)",linecolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(
                    tickfont=dict(color="rgba(220,220,220,0.75)",size=12,family="Courier New"),
                    gridcolor="rgba(255,255,255,0.07)",linecolor="rgba(255,255,255,0.1)")),
            margin=dict(l=50,r=20,t=20,b=20),
            legend=dict(font=dict(color="#aaa",size=10,family="Courier New"),
                bgcolor="rgba(10,18,32,0.85)",bordercolor="rgba(255,255,255,0.1)",borderwidth=1),
            clickmode="event")

        clicked3=st.plotly_chart(fig3,use_container_width=True,
                                  key="radar_chart",on_select="rerun",selection_mode="points")
        if clicked3 and clicked3.selection and clicked3.selection.get("points"):
            pt=clicked3.selection["points"][0]
            if pt.get("customdata"):
                st.session_state.selected=pt["customdata"]

    with col_radar_panel:
        render_panel(st.session_state.selected)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FURTHER READING
# Uses 100% native Streamlit components — no raw HTML construction at all
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.caption("ALL SOURCES FROM THE WRITE-UP")
    st.markdown("---")

    # ── 1. Research Papers ───────────────────────────────────────────────────
    st.markdown("#### 📄 &nbsp; 1. Research Papers")

    papers = [r for r in RESOURCES if r["type"] == "paper"]
    for r in papers:
        with st.container(border=True):
            col_info, col_btn = st.columns([5, 1])
            with col_info:
                if r.get("subtitle"):
                    st.caption(r["subtitle"].upper())
                st.markdown("**" + r["title"] + "**")
                st.caption(r["author"])
                if r.get("tagline"):
                    st.markdown("*" + r["tagline"] + "*")
                st.markdown(r["desc"])
            with col_btn:
                st.link_button("Read Paper ↗", r["url"], use_container_width=True)
                for lnk in r.get("extra_links", []):
                    st.link_button(lnk["label"] + " ↗", lnk["url"], use_container_width=True)

    st.markdown("---")

    # ── 2. Software & Tooling ────────────────────────────────────────────────
    st.markdown("#### 🔧 &nbsp; 2. Software & Tooling")

    tools = [r for r in RESOURCES if r["type"] == "tool"]

    # Header row
    h1, h2, h3 = st.columns([2, 3, 2])
    h1.caption("TOOL")
    h2.caption("BEST FOR...")
    h3.caption("")
    st.markdown('<hr style="margin:4px 0 8px;border-color:rgba(255,255,255,0.08);">', unsafe_allow_html=True)

    for r in tools:
        c1, c2, c3 = st.columns([2, 3, 2])
        with c1:
            st.markdown("**" + r["title"] + "**")
            st.caption(r["author"])
        with c2:
            if r.get("best_for"):
                st.markdown(r["best_for"])
            st.markdown('<div style="font-size:11px;color:rgba(180,180,200,0.5);line-height:1.6;margin-top:4px;">' + r["desc"] + '</div>', unsafe_allow_html=True)
        with c3:
            st.link_button("Visit ↗", r["url"], use_container_width=True)
            for lnk in r.get("extra_links", []):
                st.link_button(lnk["label"] + " ↗", lnk["url"], use_container_width=True)
        st.markdown('<hr style="margin:6px 0;border-color:rgba(255,255,255,0.05);">', unsafe_allow_html=True)

    st.markdown("---")

    # ── 3. Learning Paths & Communities ─────────────────────────────────────
    st.markdown("#### 🌐 &nbsp; 3. Learning Paths & Communities")

    communities = [r for r in RESOURCES if r["type"] == "community"]
    c_cols = st.columns(len(communities), gap="medium")
    for i, r in enumerate(communities):
        with c_cols[i]:
            with st.container(border=True):
                st.markdown("**" + r["title"] + "**")
                st.caption(r["author"])
                st.markdown(r["desc"])
                st.link_button("Visit ↗", r["url"], use_container_width=True)
                for lnk in r.get("extra_links", []):
                    st.link_button(lnk["label"] + " ↗", lnk["url"], use_container_width=True)

    st.markdown("---")

    # ── 4. Additional References ─────────────────────────────────────────────
    st.markdown("#### 🔗 &nbsp; 4. Additional References")

    for link in EXTRA_LINKS:
        with st.container(border=True):
            col_txt, col_btn = st.columns([5, 1])
            with col_txt:
                st.markdown("**" + link["label"] + "**")
                st.markdown(link["desc"])
                for lnk in link.get("extra_links", []):
                    st.link_button(lnk["label"] + " ↗", lnk["url"])
            with col_btn:
                st.link_button("Visit ↗", link["url"], use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:16px 0;
            border-top:1px solid rgba(255,255,255,0.06);
            font-family:'Share Tech Mono',monospace;
            font-size:10px;color:rgba(255,255,255,0.18);
            display:flex;justify-content:space-between;letter-spacing:1px;">
  <span>Mechanistic Interpretability &#8212; Resources &amp; Further Reading</span>
  <span>Knowledge Graph &middot; Timeline &middot; Radar Profile &middot; Further Reading</span>
</div>
""", unsafe_allow_html=True)
