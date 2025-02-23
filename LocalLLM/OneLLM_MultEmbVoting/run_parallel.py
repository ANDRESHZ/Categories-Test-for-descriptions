import subprocess
import multiprocessing as mp
from datetime import datetime
import os
import argparse
from tqdm import tqdm
from sys import platform
if platform == "linux" or platform == "linux2":
   # linux
   addLetter=""
elif platform == "darwin":
    # OS X
    addLetter=""
elif platform == "win32":
    # Windows...
    addLetter="X:"
RUTABASE=addLetter+"/LILO/LILO-Categories-Test/"
def run_process(params):
    """
    Execute a single process with given parameters
    
    Args:
        params (tuple): (process_id, total_processes)
    """
    process_id, total_processes = params
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Create unique output file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RUTABASE+'/LocalLLM/OneLLM_MultEmbVoting/outputs/', f'output_proc{process_id}.txt')
        
        # Construct the command
        cmd = [
            "c:/LILOTest/Scripts/python.exe",
            RUTABASE+"/LocalLLM/OneLLM_MultEmbVoting/RunOneLLM_MultEmbVoteDIVIDE.py",
            "--total_processes", str(total_processes),
            "--process_id", str(process_id)
        ]
        
        print(f"Starting process {process_id}/{total_processes-1}")
        print(f"Command: {' '.join(cmd)}")
        
        # Execute the command
        with open(output_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            return_code = process.wait()
            
        return {
            'process_id': process_id,
            'total_processes': total_processes,
            'status': 'success' if return_code == 0 else 'error',
            'output_file': output_file,
            'return_code': return_code
        }
        
    except Exception as e:
        print(f"Process {process_id} failed: {str(e)}")
        return {
            'process_id': process_id,
            'total_processes': total_processes,
            'status': 'error',
            'error': str(e)
        }

def parallel_execution(num_processes):
    """
    Execute the script in parallel
    
    Args:
        num_processes (int): Number of parallel processes to run
    """
    # Prepare parameters for each process
    process_params = [(i, num_processes) for i in range(num_processes)]
    
    # Create pool and execute processes
    print(f"Starting {num_processes} parallel processes...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = []
        with tqdm(total=num_processes, desc="Executing processes") as pbar:
            for result in pool.imap_unordered(run_process, process_params):
                results.append(result)
                pbar.update(1)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Run RunOneLLM_MultEmbVoteDIVIDE.py in parallel'
    )
    parser.add_argument(
        '--processes', 
        type=int, 
        required=True,
        help='Number of parallel processes to run'
    )
    
    args = parser.parse_args()
    
    # Record start time
    start_time = datetime.now()
    print(f"Starting parallel execution with {args.processes} processes")
    
    try:
        # Execute parallel processes
        results = parallel_execution(args.processes)
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Print results summary
        print("\nExecution Summary:")
        print(f"Total time: {execution_time}")
        
        successful = 0
        failed = 0
        
        for result in results:
            if result['status'] == 'success':
                successful += 1
                print(f"\nProcess {result['process_id']}/{result['total_processes']-1}:")
                print(f"Status: {result['status']}")
                print(f"Output file: {result['output_file']}")
                print(f"Return code: {result['return_code']}")
            else:
                failed += 1
                print(f"\nProcess {result['process_id']}/{result['total_processes']-1}:")
                print(f"Status: {result['status']}")
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        print(f"\nTotal processes: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
    except Exception as e:
        print(f"Main execution failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
