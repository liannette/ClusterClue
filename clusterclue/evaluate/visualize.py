from pathlib import Path
import logging
import csv
import subsketch as subsk
import pandas as pd
from subsketch.reports import generate_html_for_annotated_subcluster
from subsketch.io import read_compounds
from subsketch.loaders import load_mibig_bgc
import textwrap

logger = logging.getLogger(__name__)

def visualize_evaluation_results(
    annotated_subclusters_filepath,
    evaluation_best_hits_filepath,
    motif_gwms_filepath,
    ref_gbks_dirpath,
    domain_hits_filepath,
    motif_hits_filepath,
    mibig_compounds_filepath,
    out_html_dirpath
    ) -> None:

    subclusters = dict()

    with open(annotated_subclusters_filepath, "r") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            subclusters[row["subcluster_id"]] = {
                "id": row["subcluster_id"],
                "bgc_id": row["mibig_acc"],
                "compound_name": row["bgc_product"],
                "substructure_name": row["substructure"],
                "substructure_class": row["substructure class"],
                "substructure_smiles": row["substructure smiles"],
                "genes": row["genes"].split(";"),
                "protein_ids": row["protein_ids"].split(";"),
                "pathway_quality": row["pathway quality"],
                "pubmed_id": [],
                "orig_seq": "N/A",
            }

    with open(evaluation_best_hits_filepath, "r") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            subclusters[row["subcluster_id"]].update({
                "best_motif_hit": row["motif_id"],
                "overlap_score": row["overlap_score"],
                "penalized_score": row["penalized_score"],
            })

    # out_html_dirpath = motifs_dirpath / "ref_subclusters_hits"
    out_html_dirpath.mkdir(exist_ok=True, parents=True)

    session = subsk.SubSketchSession(
        motifs_file=motif_gwms_filepath,
        genbank_dir=ref_gbks_dirpath,
        domain_hits_file=domain_hits_filepath,
        motif_hits_file=motif_hits_filepath,
        compounds_file=mibig_compounds_filepath,
    )
    session.load()

    mibig_compounds = read_compounds(mibig_compounds_filepath)

    index_entries = []

    for subcluster_id, subcluster in subclusters.items():
        html_filename = f"subcluster_{subcluster_id}_hits.html"
        html_filepath = out_html_dirpath / html_filename

        bgc_id = subcluster["bgc_id"]

        bgc_filepath = Path(ref_gbks_dirpath) / f"{bgc_id}.gbk"
        bgc_data = load_mibig_bgc(bgc_filepath)

        html_content = generate_html_for_annotated_subcluster(
            subcluster=subcluster,
            bgc_data=bgc_data,
            compounds=mibig_compounds.get(bgc_id, []),
            gene_arrow_scaling=60,
            )
        html_content += session.html_report_for_bgc(
            bgc_id=bgc_id,
            include_title=False,
            include_bgc_plot=False,
            include_compound_plots=False,
            include_motif_plots=True,
            gene_arrow_scaling=60, 
        )

        with open(html_filepath, "w") as outfile:
            outfile.write(html_content)

        motif_hits = session.data.bgc2hits.get(bgc_id, [])
        motif_ids = [hit["motif_id"] for hit in motif_hits]

        index_entries.append(
            {
                "href": html_filename,
                "subcluster_id": subcluster["id"],
                "substructure_name": subcluster["substructure_name"],
                "substructure_class": subcluster["substructure_class"],
                "best_motif_hit": subcluster.get("best_motif_hit", "N/A"),
                "overlap_score": subcluster.get("overlap_score", "N/A"),
                "penalized_score": subcluster.get("penalized_score", "N/A"),
                "bgc_id": bgc_id,
                "motif_ids": motif_ids
            }
        )

    index_html_path = out_html_dirpath / "index.html"
    write_index_html(index_html_path, index_entries)
    logger.info(f"Generated index.html with {len(index_entries)} subcluster visualizations at {index_html_path}")


def write_index_html(index_html_path, index_entries):
    """Write a beautiful index.html with links to subcluster HTML files."""
    
    with open(index_html_path, "w") as f:
        # Header
        f.write(textwrap.dedent(f"""\
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset='utf-8'>
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Subcluster Visualizations</title>
              <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                
                body {{
                  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                  line-height: 1.6; color: #333;
                  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  min-height: 100vh; padding: 2rem;
                }}
                
                .container {{
                  max-width: 1400px; margin: 0 auto;
                  background: rgba(255, 255, 255, 0.95);
                  backdrop-filter: blur(10px); border-radius: 20px;
                  box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden;
                }}
                
                header {{
                  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                  color: white; padding: 2.5rem; text-align: center;
                }}
                
                h1 {{ font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; 
                     text-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
                
                .subtitle {{ opacity: 0.9; font-size: 1.1rem; }}
                
                .stats {{ display: flex; justify-content: center; gap: 2rem; 
                         margin-top: 1rem; flex-wrap: wrap; }}
                
                .stat {{ text-align: center; background: rgba(255,255,255,0.2);
                        padding: 1rem 1.5rem; border-radius: 12px; 
                        backdrop-filter: blur(5px); }}
                
                .stat-number {{ font-size: 1.8rem; font-weight: 800; display: block; }}
                
                main {{ padding: 2.5rem; }}
                
                .table-container {{ overflow-x: auto; border-radius: 16px;
                                   box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
                                   margin-top: 2rem; }}
                
                table {{ width: 100%; border-collapse: collapse; background: white;
                        border-radius: 16px; overflow: hidden; }}
                
                th {{ background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                     padding: 1.25rem 1.5rem; text-align: left; font-weight: 600;
                     font-size: 0.95rem; text-transform: uppercase; 
                     letter-spacing: 0.05em; color: #475569; 
                     border-bottom: 3px solid #e2e8f0; }}
                
                td {{ padding: 1.25rem 1.5rem; border-bottom: 1px solid #f1f5f9;
                     vertical-align: middle; }}
                
                tr:hover {{ background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
                           transform: scale(1.01); transition: all 0.2s ease; }}
                
                .sc-link {{ color: #4f46e5; text-decoration: none; font-weight: 600;
                            padding: 0.5rem 0.75rem; border-radius: 8px;
                            transition: all 0.2s ease; display: inline-flex;
                            align-items: center; gap: 0.5rem; }}
                
                .sc-link:hover {{ background: #4f46e5; color: white;
                                  transform: translateY(-1px);
                                  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4); }}
                
                .sc-link::before {{ content: '🔗'; font-size: 1.1em; }}
                
                .motif-ids {{ background: #f8fafc; padding: 0.75rem; border-radius: 8px;
                             font-family: 'Monaco', 'Menlo', monospace; font-size: 0.9rem;
                             border-left: 4px solid #4f46e5; word-break: break-all; }}
                
                @media (max-width: 768px) {{
                  body {{ padding: 1rem; }}
                  h1 {{ font-size: 2rem; }}
                  main {{ padding: 1.5rem; }}
                  th, td {{ padding: 1rem 0.75rem; }}
                  .stats {{ gap: 1rem; }}
                }}
              </style>
            </head>
            <body>
              <div class="container">
                <header>
                  <h1>Subcluster Visualizations</h1>
                  <p class="subtitle">Overview of detected subcluster hits</p>
                  <div class="stats">
                    <div class="stat">
                      <span class="stat-number">{len(index_entries)}</span>
                      Total Subclusters
                    </div>
                  </div>
                </header>
                <main>
                  <div class="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Subcluster ID</th>
                          <th>Substructure name</th>
                          <th>Substructure class</th>
                          <th>Best Motif Hit</th>
                          <th>Overlap Score</th>
                          <th>Penalized Score</th>
                          <th>MiBIG accession</th>
                          <th>Detected motif IDs</th>
                        </tr>
                      </thead>
                      <tbody>"""))
        
        # Table rows - matching your index_entries order
        for entry in index_entries:
            motif_ids_str = ", ".join(entry["motif_ids"]) if entry["motif_ids"] else "None"
            f.write(textwrap.dedent(f"""\
                <tr>
                  <td><a href="{entry['href']}" class="sc-link">{entry['subcluster_id']}</a></td>
                  <td>{entry['substructure_name'] or 'N/A'}</td>
                  <td>{entry['substructure_class'] or 'N/A'}</td>
                  <td>{entry['best_motif_hit'] or 'N/A'}</td>
                  <td>{entry['overlap_score'] or 'N/A'}</td>
                  <td>{entry['penalized_score'] or 'N/A'}</td>
                  <td>{entry['bgc_id'] or 'N/A'}</td>
                  <td><div class="motif-ids">{motif_ids_str}</div></td>
                </tr>"""))
        
        # Footer
        f.write(textwrap.dedent("""\
                      </tbody>
                    </table>
                  </div>
                </main>
              </div>
            </body>
            </html>"""))
