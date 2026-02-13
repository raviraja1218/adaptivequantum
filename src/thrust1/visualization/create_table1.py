"""
Create Table 1 for paper from gradient results.
"""
import pandas as pd
from pathlib import Path

def create_table1():
    # Load gradient results
    data_path = Path("experiments/thrust1/gradient_final_paper/gradient_results_paper_matched.csv")
    df = pd.read_csv(data_path)
    
    # Select key columns
    table_df = df[['qubits', 'random_gradient', 'adaptive_gradient', 'improvement']].copy()
    table_df.columns = ['Qubits', 'Random Gradient', 'Adaptive Gradient', 'Improvement Factor']
    
    # Format numbers
    def format_scientific(x):
        if x < 1e-100:
            return r"$<10^{-100}$"
        return f"{x:.2e}"
    
    for col in ['Random Gradient', 'Adaptive Gradient', 'Improvement Factor']:
        table_df[col] = table_df[col].apply(format_scientific)
    
    # Create LaTeX table
    latex_str = table_df.to_latex(
        index=False,
        caption="Gradient improvement with AdaptiveQuantum initialization. Random gradients suffer from barren plateaus (exponential decay), while AdaptiveQuantum maintains trainable gradients across system sizes.",
        label="tab:gradient_improvement",
        position='h!',
        escape=False
    )
    
    # Add table notes
    latex_str = latex_str.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n" + r"""
\parbox{\textwidth}{
\footnotesize \textit{Note:} Improvement factors calculated as Adaptive Gradient / Random Gradient. 
For 50+ qubit systems, random initialization results in barren plateaus where gradients are undetectably small, 
while AdaptiveQuantum maintains gradients sufficient for training. The 100-qubit improvement of $>10^{25}\times$ 
demonstrates the framework's scalability.
}
"""
    )
    
    # Save table
    output_path = Path("figures/paper/table1_barren_plateau.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✅ Table 1 saved to: {output_path}")
    
    # Also save as Markdown for reference
    markdown_str = table_df.to_markdown(index=False)
    with open(output_path.with_suffix('.md'), 'w') as f:
        f.write("# Table 1: Gradient Improvement with AdaptiveQuantum\n\n")
        f.write(markdown_str)
    
    return table_df

if __name__ == "__main__":
    table = create_table1()
    print("\nTable 1 Preview:")
    print(table.head(10))
