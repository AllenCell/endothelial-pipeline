import os

def get_output_path(workflow_name: str, verbose:bool=True) -> str:
    """
    Save results to a universal results directory in a folder titled after the workflow. 
    The contents are gitignored. 
    """
    repo_top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    output_dir = os.path.join(repo_top_dir, 'results', workflow_name) + os.sep
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if verbose:
        print(f"Output saved to directory: {output_dir}")
    
    return output_dir
    