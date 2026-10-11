import docx
from pathlib import Path
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_callout(doc, title, text, bg_hex="EBF3FA", border_hex="1B365D"):
    """Adds a stylish callout box for highlighting key features like CPU/Non-GPU execution."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    # Left border only (accent line)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'  <w:top w:val="none"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    r_title = p.add_run(title + "\n")
    r_title.font.name = 'Calibri'
    r_title.font.bold = True
    r_title.font.size = Pt(11)
    r_title.font.color.rgb = RGBColor(27, 54, 93)
    
    r_text = p.add_run(text)
    r_text.font.name = 'Calibri'
    r_text.font.size = Pt(10)
    r_text.font.color.rgb = RGBColor(40, 40, 40)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(4)
    p_spacer.paragraph_format.space_after = Pt(4)

def create_document():
    doc = docx.Document()
    
    # Page setup - Margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles definition
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(40, 40, 40)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    NAVY = RGBColor(27, 54, 93)       # #1B365D
    SLATE = RGBColor(70, 80, 95)      # #46505F
    DARK = RGBColor(30, 30, 30)
    EMPHASIS = RGBColor(0, 102, 153)  # #006699

    # Pre-title
    p_pre = doc.add_paragraph()
    p_pre.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_pre.paragraph_format.space_after = Pt(2)
    run_pre = p_pre.add_run("FINAL YEAR PROJECT SYNOPSIS")
    run_pre.font.name = 'Calibri'
    run_pre.font.size = Pt(13)
    run_pre.font.bold = True
    run_pre.font.color.rgb = SLATE

    # Main Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("ENTERPRISE GRAPHRAG SYSTEM")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(12)
    run_sub = p_sub.add_run("An Offline, Privacy-Preserving Knowledge Graph-Augmented Retrieval System with Hybrid Vector-Graph Indexing and Gemma 4 LLM (Optimized for Non-GPU & CPU-Only Commodity Systems)")
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(11.5)
    run_sub.font.italic = True
    run_sub.font.color.rgb = SLATE

    # Highlight Banner Box
    add_callout(
        doc,
        "CORE ARCHITECTURAL HIGHLIGHT: 100% NON-GPU & CPU-ONLY COMPATIBLE",
        "Unlike conventional GraphRAG and enterprise LLM solutions that necessitate expensive multi-GPU server infrastructure, this system is specifically engineered from the ground up to run autonomously on standard non-GPU, CPU-only commodity workstations and laptops. Utilizing the state-of-the-art Google Gemma 4 model (compact 1B/e4b architecture), CPU-optimized FAISS vector indexing (faiss-cpu), and multi-tier fallbacks, full enterprise document extraction and conversational intelligence are democratized on standard corporate hardware without any dedicated AI accelerator cards.",
        bg_hex="F0F7FA",
        border_hex="006699"
    )

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = NAVY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(12.5)
        run.font.bold = True
        run.font.color.rgb = SLATE
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r_bold = p.add_run(bold_prefix)
        r_bold.font.bold = True
        r_bold.font.color.rgb = DARK
        r_text = p.add_run(text)
        r_text.font.color.rgb = DARK
        return p

    # -------------------------------------------------------------
    # 1. Topic of the Project
    # -------------------------------------------------------------
    add_h1("1. Topic of the Project")
    p = doc.add_paragraph()
    r = p.add_run("Enterprise GraphRAG: An Offline, Privacy-Preserving Knowledge Graph-Augmented Retrieval System with Hybrid Vector-Graph Indexing and Gemma 4 LLM Optimized for Non-GPU Commodity Workstations")
    r.font.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = NAVY

    p_topic_exp = doc.add_paragraph()
    p_topic_exp.add_run(
        "The project title corroborates the synthesis of Knowledge Graphs (KG) and Retrieval-Augmented Generation (RAG), "
        "emphasizing complete air-gapped privacy and hardware democratization. Crucially, the topic underscores the system's "
        "inherent capability to execute on non-GPU based systems (pure CPU hardware) using the compact, high-reasoning Google Gemma 4 model family, "
        "eliminating enterprise reliance on high-cost cloud APIs or specialized AI accelerators."
    )

    # -------------------------------------------------------------
    # 2. Introduction of the Project
    # -------------------------------------------------------------
    add_h1("2. Introduction of the Project")
    doc.add_paragraph(
        "In modern enterprises, mission-critical knowledge is predominantly locked inside unstructured, heterogeneous documents—ranging "
        "from scanned agreements, invoices, and standard operating procedures to engineering manuals and compliance audits. "
        "Existing Information Retrieval (IR) and standard Vector-based RAG architectures suffer from three critical bottlenecks:\n"
        "1. Topological Blindness: Vector chunk embeddings cannot cross-reference multi-hop entities or aggregate knowledge dispersed across hundreds of distinct files.\n"
        "2. Hallucinations & Verifiable Audit Gaps: Standard LLM outputs lack deterministic relational grounding, generating unverifiable claims.\n"
        "3. Prohibitive Hardware & Cloud Costs: Conventional high-performance RAG pipelines either require leasing high-cost enterprise cloud endpoints (violating HIPAA/GDPR data residency) or purchasing costly enterprise GPU clusters (NVIDIA A100/H100) that exceed commodity department budgets."
    )

    doc.add_paragraph(
        "To overcome these limitations, the Enterprise GraphRAG System provides a production-grade, 100% offline knowledge platform "
        "purpose-built to operate seamlessly on non-GPU based systems (pure CPU) as well as GPU-accelerated environments. "
        "By fusing dense vector retrieval (FAISS-CPU) with native property graph traversals (Neo4j), topological graph algorithms (NetworkX), "
        "and a high-efficiency local generation engine featuring Google Gemma 4 (specifically google/gemma-4-e4b-it / Gemma 4 1B family), "
        "the application achieves state-of-the-art document intelligence directly on standard corporate PC hardware."
    )

    # -------------------------------------------------------------
    # 3. Objective and Scope
    # -------------------------------------------------------------
    add_h1("3. Objective and Scope")
    add_h2("3.1 Core Objectives")
    add_bullet("Native Non-GPU & CPU-Only Execution: ", "Deliver full operational autonomy on machines without dedicated graphics cards, utilizing CPU thread optimization, low-memory model loading (low_cpu_mem_usage=True), and native Ollama / llama.cpp / Transformers CPU execution.")
    add_bullet("Integration of High-Efficiency Gemma 4 LLM: ", "Incorporate Google's next-generation Gemma 4 architecture (1B parameter class), offering an optimal balance of low parameter footprint, high reasoning density, and low CPU RAM requirements (< 2–3 GB memory footprint).")
    add_bullet("Zero Cloud Exfiltration (Air-Gapped Compliance): ", "Guarantee 100% on-premises execution where OCR, embeddings, graph operations, and generation run entirely offline, meeting stringent compliance mandates.")
    add_bullet("Dual-Store Hybrid Indexing: ", "Simultaneously index chunk embeddings into FAISS-CPU (IVFFlat inner-product metric) and extract typed entity-relation graphs into a Neo4j Property Graph.")
    add_bullet("Computer-Vision Assisted Ingestion & OCR: ", "Process PDF, DOCX, TXT, MD, and HTML files via RapidOCR (ONNX Runtime CPU), PyMuPDF, Tesseract, and OpenCV deskewing/denoising, reconstructing tables and hierarchical layouts without requiring GPU acceleration.")
    add_bullet("Multi-Hop Relational Retrieval & Deduplication: ", "Fuse dense semantic similarity with multi-hop Cypher graph expansions using reciprocal fusion scoring and Maximal Marginal Relevance (MMR) deduplication.")
    add_bullet("Evidence-Grounded Answers with Inline Citations: ", "Synthesize answers bound strictly to retrieved context, with deterministic inline citations ([Doc: Page: Chunk]) to completely eliminate hallucination.")
    add_bullet("Interactive Topology Visualization: ", "Provide a responsive Single Page Application (SPA) with a real-time Cytoscape.js canvas for interactive graph clustering, community inspection, and document isolation.")

    add_h2("3.2 Scope and End-User Impact")
    doc.add_paragraph(
        "The project culminates in a self-contained, turnkey enterprise desktop and intranet software application. "
        "By removing the GPU hardware barrier, the system empowers a broad spectrum of real-world enterprise users:"
    )
    add_bullet("Standard Corporate Offices & SMBs: ", "Run state-of-the-art document intelligence on existing office desktops (Intel Core i5/i7 or AMD Ryzen with 16 GB RAM) without capital expenditure on GPU infrastructure.")
    add_bullet("Legal & Compliance Teams: ", "Analyze massive multi-contract archives and perform compliance cross-checks with zero risk of proprietary data exposure to external cloud providers.")
    add_bullet("Air-Gapped & Defense Environments: ", "Deploy fully functional GraphRAG intelligence in secure, physically disconnected field laptops and air-gapped servers lacking discrete GPUs.")

    # -------------------------------------------------------------
    # 4. Process Description
    # -------------------------------------------------------------
    add_h1("4. Process Description")
    doc.add_paragraph(
        "The software architecture is decoupled into four continuous, asynchronously managed pipelines: "
        "(1) Multi-Engine Ingestion & Layout Reconstruction, (2) Dual Graph-Vector Indexing, (3) Hybrid Multi-Hop Retrieval, and (4) Context-Grounded Answer Generation."
    )

    add_h2("4.1 Process Architecture & Hardware Adaptation")
    
    # Architecture Table
    arch_table = doc.add_table(rows=5, cols=2)
    arch_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(arch_table, color="1B365D", sz="8")
    
    stages = [
        ("Stage 1: Document Ingestion & OCR", "Accepts diverse formats (PDF, DOCX, TXT, HTML). Classifies documents into digital vs. scanned. Preprocesses images using OpenCV (deskewing, binarization) and extracts text/tables via RapidOCR (CPU ONNX runtime) and PyMuPDF, segmenting text via recursive character chunking."),
        ("Stage 2: Dual Indexing (Graph + Vector)", "Generates 384-dimensional dense vectors using SentenceTransformers (all-MiniLM-L6-v2 running on CPU/GPU) for FAISS-CPU IVFFlat indexing. Concurrently executes Gemma 4 prompt templates to extract structured entities and typed relations, merging them into Neo4j."),
        ("Stage 3: Hybrid Retrieval & Fusion", "Classifies query intent and decomposes compound queries. Queries FAISS-CPU for top-K semantic chunks and executes Cypher graph traversals (neighborhood expansion and shortest paths). Ranks candidates using weighted reciprocal scoring and MMR deduplication."),
        ("Stage 4: Gemma 4 Grounded Generation", "Constructs a grounded system prompt combining relational knowledge triples and prioritized text chunks. Gemma 4 (operating on CPU or GPU) generates a coherent, context-grounded response with deterministic inline citations."),
        ("Stage 5: Visualization & Telemetry", "Renders interactive knowledge graphs in Cytoscape.js with community grouping, while the telemetry dashboard provides real-time CPU thread, RAM, and index status monitoring.")
    ]
    for idx, (title, desc) in enumerate(stages):
        cell_t = arch_table.rows[idx].cells[0]
        cell_d = arch_table.rows[idx].cells[1]
        cell_t.width = Inches(2.2)
        cell_d.width = Inches(4.3)
        set_cell_margins(cell_t, top=80, bottom=80, left=100, right=100)
        set_cell_margins(cell_d, top=80, bottom=80, left=100, right=100)
        
        p_t = cell_t.paragraphs[0]
        r_t = p_t.add_run(title)
        r_t.font.bold = True
        r_t.font.size = Pt(10)
        r_t.font.color.rgb = NAVY
        set_cell_background(cell_t, "F0F4F8")

        p_d = cell_d.paragraphs[0]
        r_d = p_d.add_run(desc)
        r_d.font.size = Pt(9.5)
        r_d.font.color.rgb = DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_h2("4.2 Hardware Abstraction & CPU Execution Architecture")
    doc.add_paragraph(
        "A standout engineering aspect of the system is its intelligent Hardware Abstraction Layer implemented in utils.py, generation.py, and embedding.py:\n"
        "• Automatic Device Detection: The system dynamically probes for CUDA availability via is_cuda_available(). If a dedicated GPU is absent, it seamlessly defaults to CPU execution without throwing runtime faults or configuration errors.\n"
        "• Gemma 4 CPU Optimization: When running on CPU, the generation subsystem disables GPU-only bitsandbytes int8 CUDA kernels, loading Gemma 4 into system RAM using PyTorch low_cpu_mem_usage=True. Alternatively, it natively interfaces with local Ollama CPU instances (utilizing highly optimized AVX2/AVX-512 CPU instructions), yielding fast, responsive generation on standard multi-core processors.\n"
        "• Zero-Dependency Fallback in Embeddings: If deep-learning libraries encounter native DLL issues on stripped-down CPU hosts, the embedding engine activates an internal deterministic hash embedding mode (384-dimensional MD5 n-gram hashing), guaranteeing 100% operational uptime."
    )

    # -------------------------------------------------------------
    # 5. Resources and Limitations
    # -------------------------------------------------------------
    add_h1("5. Resources and Limitations")
    add_h2("5.1 Hardware and Software Resource Requirements")

    res_table = doc.add_table(rows=1, cols=3)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(res_table, color="B0C4DE", sz="6")

    headers = ["Resource Category", "Component / Tool", "Technical Function & Purpose"]
    hdr_cells = res_table.rows[0].cells
    for i, h in enumerate(headers):
        p = hdr_cells[i].paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)

    rows_data = [
        ("Base Hardware (Non-GPU)", "Standard Multi-Core CPU (Intel Core i5/i7, AMD Ryzen 5/7)", "PRIMARY REQUIREMENT: Zero dedicated GPU required. Executes all pipelines on standard CPU threads."),
        ("Optional GPU Acceleration", "NVIDIA RTX Series (CUDA 12.x)", "OPTIONAL: If present, automatically utilized for accelerated tensor calculations."),
        ("System Memory (RAM)", "16 GB DDR4 / DDR5 RAM", "Accommodates Gemma 4 model (~2–3 GB RAM), Neo4j JVM heap (~1–2 GB), FAISS index, and OS buffers."),
        ("Local Storage", "15–20 GB Free SSD Space", "Houses local model checkpoints, Neo4j graph store, SQLite database, and document uploads."),
        ("Host Operating System", "Windows 11 / Windows 10 (64-bit) or Linux (Ubuntu 22.04+)", "Standard commercial operating system with Python 3.11 runtime."),
        ("Generative LLM", "Google Gemma 4 (gemma-4-e4b-it / Gemma 4 1B)", "High-efficiency local model running on CPU (Transformers / Ollama) for extraction and synthesis."),
        ("Dense Vector Embedder", "SentenceTransformers (all-MiniLM-L6-v2)", "Compact 384-dimensional semantic embedding model optimized for CPU inference."),
        ("Vector Search Engine", "FAISS-CPU (v1.9.0)", "High-performance inner-product similarity vector index executing purely on CPU."),
        ("Property Graph Engine", "Neo4j Community Edition 5.27+", "Local property graph database running on host JVM for Cypher relationship traversals."),
        ("Topological Processing", "NetworkX 3.4.2", "CPU-based community detection (Louvain), shortest-path analytics, and centrality scoring."),
        ("OCR & Image Pipeline", "RapidOCR (ONNX CPU), PyMuPDF, OpenCV, Tesseract", "Complete non-GPU computer vision pipeline for scanned document conversion and deskewing."),
        ("Relational & Cache Store", "SQLite 3 via SQLAlchemy 2.0 ORM", "Persistent document catalog, chunk mapping, ingestion job state, and query cache."),
        ("Web Framework & UI", "FastAPI, Uvicorn, Cytoscape.js, Modern CSS/JS", "Async backend server and interactive browser-based visualization dashboard.")
    ]

    for cat, comp, func in rows_data:
        row = res_table.add_row()
        c0, c1, c2 = row.cells
        c0.width = Inches(1.8)
        c1.width = Inches(2.2)
        c2.width = Inches(2.5)
        
        for c in (c0, c1, c2):
            set_cell_margins(c, top=60, bottom=60, left=100, right=100)
            set_cell_background(c, "FBFDFF")

        r0 = c0.paragraphs[0].add_run(cat)
        r0.font.bold = True
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = NAVY

        r1 = c1.paragraphs[0].add_run(comp)
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = DARK

        r2 = c2.paragraphs[0].add_run(func)
        r2.font.size = Pt(9)
        r2.font.color.rgb = DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_h2("5.2 Practical Limitations of the Proposed System")
    add_bullet("1. CPU Generation Throughput: ", "While Gemma 4 runs efficiently on CPU, token generation speeds on CPU (~8–18 tokens/sec depending on processor core count and AVX support) are naturally lower than dedicated high-end GPU cards. However, for document analysis and Q&A, this speed is fully viable for interactive use.")
    add_bullet("2. Deep Ingestion Extraction Latency: ", "Performing multi-pass OCR, table detection, and LLM entity extraction across multi-thousand-page repositories on CPU requires batched or background scheduling compared to instantaneous superficial chunking.")
    add_bullet("3. Bounded Context Window: ", "The generation context window is capped at 3,072–4,096 tokens to ensure low RAM consumption and fast CPU prompt evaluation, mitigated by aggressive MMR context pruning.")
    add_bullet("4. Standalone Graph Instance: ", "Neo4j Community Edition operates as a single standalone server instance, tailored for desktop and local workstation environments rather than globally distributed server clusters.")

    # -------------------------------------------------------------
    # 6. Conclusion
    # -------------------------------------------------------------
    add_h1("6. Conclusion")
    doc.add_paragraph(
        "The Enterprise GraphRAG System demonstrates that advanced, knowledge-graph-augmented conversational document intelligence "
        "does not require expensive enterprise GPU infrastructure or cloud vendor lock-in. By coupling the high parameter efficiency of "
        "Google Gemma 4 with FAISS-CPU indexing, local Neo4j graph storage, and a robust multi-engine OCR pipeline, the project "
        "democratizes enterprise-grade GraphRAG on everyday non-GPU commodity hardware."
    )

    doc.add_paragraph(
        "Key Standout Achievements and Competitive Advantages:"
    )
    add_bullet("Universal Hardware Compatibility (Non-GPU & CPU-First): ", "The system runs out-of-the-box on standard desktop CPUs without requiring specialized NVIDIA cards, dramatically lowering deployment costs for academic, governmental, and corporate institutions.")
    add_bullet("Harnessing Google Gemma 4: ", "Leverages the superior architectural efficiency of Gemma 4, delivering exceptional factual extraction and synthesis while occupying only ~2–3 GB of system RAM.")
    add_bullet("Absolute Air-Gapped Data Confidentiality: ", "Zero external API calls ensure 100% compliance with strict privacy, security, and IP governance regulations.")
    add_bullet("Dual-Channel Hybrid Fusion: ", "Combines semantic vector matching with multi-hop topological graph traversals, eliminating relational blind spots present in traditional RAG.")
    add_bullet("Zero Hallucination with Verifiable Grounding: ", "Enforces strict inline citations linking every generative statement directly back to source documents, page numbers, and chunk references.")
    add_bullet("Human-in-the-Loop Topological Visualization: ", "Transforms opaque document archives into an interactive, visually inspectable knowledge graph using Cytoscape.js.")

    # Save to Target Path
    target_path = Path(r"c:\Users\Loq\Documents\programs\final year project\graph\Project_Synopsis.docx")
    try:
        doc.save(str(target_path))
        print(f"Successfully updated and saved: {target_path}")
    except PermissionError:
        alt_path = Path(r"c:\Users\Loq\Documents\programs\final year project\graph\Project_Synopsis_Updated.docx")
        doc.save(str(alt_path))
        print(f"Project_Synopsis.docx is currently open in Microsoft Word. Saved updated file to: {alt_path}")

if __name__ == "__main__":
    create_document()
