import argparse
import ast
import json
import warnings
from tqdm import tqdm
from transformers import pipeline, RobertaTokenizer, EncoderDecoderModel

# Suppress a specific warning from the AST unparser for cleaner output
warnings.filterwarnings("ignore", category=ast.Constant)

class StatementExtractor(ast.NodeVisitor):
    """
    An AST visitor to extract individual statements from a Python code snippet.
    """
    def __init__(self):
        self.statements = []

    def visit(self, node):
        # We only want to process the body of the first function or class definition
        if isinstance(node, ast.Module) and node.body:
            # We assume the main logic is in the first top-level function/class
            main_node = node.body[0]
            if isinstance(main_node, (ast.FunctionDef, ast.ClassDef)):
                for sub_node in main_node.body:
                    super().visit(sub_node)
            else: # If not a function/class, visit all top-level nodes
                super().visit(node)
        else:
            super().visit(node)
            
    def generic_visit(self, node):
        # This is the handler for simple, non-compound statements
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            self.statements.append(ast.unparse(node).strip())

    # --- Special handling for compound statements ---

    def visit_If(self, node):
        self.statements.append(f"if {ast.unparse(node.test)}:")
        for sub_node in node.body:
            self.visit(sub_node)
        if node.orelse:
            # A simple "else:" is an empty list, a block has nodes.
            self.statements.append("else:")
            for sub_node in node.orelse:
                self.visit(sub_node)

    def visit_For(self, node):
        self.statements.append(f"for {ast.unparse(node.target)} in {ast.unparse(node.iter)}:")
        for sub_node in node.body:
            self.visit(sub_node)

    def visit_While(self, node):
        self.statements.append(f"while {ast.unparse(node.test)}:")
        for sub_node in node.body:
            self.visit(sub_node)

    def visit_With(self, node):
        # Reconstruct the "with" line from its items
        items = ', '.join([ast.unparse(item) for item in node.items])
        self.statements.append(f"with {items}:")
        for sub_node in node.body:
            self.visit(sub_node)

    def visit_Try(self, node):
        self.statements.append("try:")
        for sub_node in node.body:
            self.visit(sub_node)
        for handler in node.handlers:
            if handler.type:
                self.statements.append(f"except {ast.unparse(handler.type)}" + (f" as {handler.name}:" if handler.name else ":"))
            else:
                self.statements.append("except:")
            for sub_node in handler.body:
                self.visit(sub_node)

def segment_code(source_code: str):
    """
    Splits a source code string into a list of statements.
    It first tries to use AST parsing and falls back to line splitting if parsing fails.
    """
    try:
        # The 'solution' from evaluations.jsonl often contains the full test script.
        # We need to extract just the function definition for clean parsing.
        tree = ast.parse(source_code)
        # Find the first function or class to parse, assuming it's the solution
        solution_node = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                solution_node = node
                break
        
        if solution_node:
             # Re-parse just the single function/class to isolate it
            tree = ast.parse(ast.unparse(solution_node))
        
        extractor = StatementExtractor()
        extractor.visit(tree)
        return extractor.statements
    except SyntaxError:
        # Fallback for non-compilable code, as described in the paper
        return [line.strip() for line in source_code.split('\n') if line.strip()]

def process_file(args):
    """
    Main function to process the generated code, generate comments, and save the output.
    """
    # 1. Load the fine-tuned comment generation model
    print(f"Loading comment generation model from: {args.model_path}...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    model = EncoderDecoderModel.from_pretrained(args.model_path)
    comment_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=args.device)
    print("Model loaded successfully.")

    # 2. Process the input file
    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
        # Read all lines at once to process in a batch for efficiency
        lines = infile.readlines()
        
        for line in tqdm(lines, desc="Processing Code Snippets"):
            data = json.loads(line)
            code_snippet = data.get("solution", "")
            
            if not code_snippet:
                continue

            # 3. Split code into statements
            statements = segment_code(code_snippet)
            
            if not statements:
                continue
            
            # 4. Generate comments for all statements in a batch
            generated_comments = comment_generator(statements, max_length=128, num_beams=4)
            comments = [result['generated_text'] for result in generated_comments]
            
            # 5. Construct the output JSON object
            code_with_comments = []
            for stmt, comment in zip(statements, comments):
                code_with_comments.append({
                    "statement": stmt,
                    "comment": comment
                })
            
            # Build the new JSON object, preserving original data and adding new fields
            output_data = {
                "task_id": data["task_id"],
                "completion_id": data["completion_id"],
                "benchmark": args.benchmark,
                "iteration": args.iteration,
                "initial_model": args.initial_model,
                "initial_code_snippet": code_snippet,
                "initial_passed": data["passed"],
                "comment_generation_model": args.model_path,
                "code_with_comments": code_with_comments,
                "annotator_feedback": [] # Placeholder for the next step
            }
            
            outfile.write(json.dumps(output_data) + '\n')

    print(f"\nProcessing complete. Output saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comments for code snippets.")
    # Input/Output files
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input `evaluations.jsonl` file from OpenCodeEval.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")

    # Model and device
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned comment generation model directory.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for model inference ('cuda:0', 'cpu', etc.).")

    # Metadata arguments
    parser.add_argument("--benchmark", type=str, required=True, help="Name of the benchmark (e.g., 'HumanEval', 'MBPP').")
    parser.add_argument("--benchmark_data_file", type=str, required=True, help="Path to the original benchmark data file (e.g., HumanEval.jsonl) to retrieve task descriptions.")
    parser.add_argument("--initial_model", type=str, required=True, help="Name of the model that generated the initial code snippets.")
    parser.add_argument("--iteration", type=int, default=1, help="The iteration number for this round of feedback.")
    
    args = parser.parse_args()
    process_file(args)