import os
import pandas as pd
import pyodbc
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay  # Business Day offset
import warnings

# Suppress the specific pandas warning about DBAPI connections
warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy")

# Configuration settings - all hardcoded instead of reading from Excel
# Benchmark mappings: (bm_code, bbg_name, holdings_query)
BENCHMARKS = [
    ("DEX_UPM", "FTSE_CANADA_UNIVERSE", "bm.pc_bond_par_value / 1000"),
    ("JPM_GBIG", "JPM_GBI_BM", "bm.mkt_value"),
    # ("JPM_EMBI", "EMBI_PLUS", "bm.mkt_value"),
    ("JPM_GBI-EM GLBL", "JPM_GBI_EM", "bm.mkt_value")
]

# Directory paths - production path is commented out for easy switching later
# Original production directory
BLOOMBERG_UPLOAD_DIR = r"\\CI_MAIN\Capmgmt\Signature\FI Rates\Bloomberg Uploads\Benchmark Uploads"

def connect_to_db():
    """Create connection to the database using the successful connection pattern"""
    # Using the connection string format from your example
    conn_str = 'Driver={SQL Server}; Server=pm-analytics-prd.ci.aws,1625;Database=PMAR_IWMS;Trusted_Connection=Yes'
    return pyodbc.connect(conn_str)

def run_query(query):
    """Execute a query and return results as DataFrame"""
    try:
        conn = connect_to_db()
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Database query error: {e}")
        # Return empty DataFrame with expected columns to prevent further errors
        return pd.DataFrame(columns=["ID_ISIN", "Holdings", "Fund", "Date", "Price"])

def calculate_dates():
    """Calculate end_date and start_date based on Excel formulas
    end_date: =WORKDAY(TODAY(),-1)  # Previous business day
    start_date: =DATE(YEAR(G1),MONTH(G1),DAY(G1)-15)  # End date minus 15 days
    """
    # Get current date
    today = pd.Timestamp.today()
    
    # Calculate end_date (previous business day)
    end_date = today - BDay(1)
    
    # Calculate start_date (end_date minus 15 days)
    start_date = end_date - timedelta(days=15)
    
    return end_date, start_date

def get_bond_data(bench_code, bench_name, holdings_query, start_dt, end_dt):
    """Get bond data for a specific benchmark and save to Excel"""
    # Create file name
    directory = BLOOMBERG_UPLOAD_DIR
    
    # Make sure directory exists
    os.makedirs(directory, exist_ok=True)
    
    xlsfilename = os.path.join(directory, f"{bench_name}.xlsx")
    
    # Create SQL query
    query = f"""SELECT COALESCE(sm.isin, sm.cusip, sm.symbol, sm.bbgid, bm.instr_code), 
                {holdings_query}, '{bench_name}', bm.as_of_date, bm.price 
                FROM [PMAR_IWMS].dbo.bm_positions bm 
                LEFT JOIN [PMAR_IWMS].dbo.ta_smf sm ON bm.instr_code = sm.instr_code 
                WHERE as_of_date >= '{start_dt}' AND as_of_date <= '{end_dt}' 
                AND bm_code = '{bench_code}'
                ORDER BY as_of_date"""
    
    # print(query)
    # Run query
    df = run_query(query)
    
    # Log how many records were pulled
    record_count = len(df)
    print(f"Benchmark {bench_name}: Retrieved {record_count} records")
    
    # Check if dataframe is empty
    if df.empty:
        print(f"Warning: No data returned for benchmark {bench_name}")
        if os.path.exists(xlsfilename):
            print(f"Keeping existing file: {xlsfilename}")
        else:
            print(f"No existing file found for benchmark {bench_name}")
        return None
    
    # Only delete the existing file if we have data to replace it with
    if os.path.exists(xlsfilename):
        os.remove(xlsfilename)
        print(f"Deleted existing file: {xlsfilename}")
    
    # Rename columns
    df.columns = ["ID_ISIN", "Holdings", "Fund", "Date", "Price"]
    
    # Convert date format
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%m/%d/%Y")
    
    # Save to Excel
    try:
        with pd.ExcelWriter(xlsfilename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
        print(f"Created file: {xlsfilename} with {record_count} records")
        return xlsfilename
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        return None

def run_benchmarks():
    """Process all benchmarks using hardcoded values"""
    
    # Calculate dates
    end_dt, start_dt = calculate_dates()
    
    # Format dates for SQL query
    start_dt_str = start_dt.strftime('%Y-%m-%d')
    end_dt_str = end_dt.strftime('%Y-%m-%d')
    
    print(f"Processing benchmarks from {start_dt_str} to {end_dt_str}")
    print(f"Output directory: {BLOOMBERG_UPLOAD_DIR}")
    
    # Process each benchmark
    successful_benchmarks = 0
    total_records_processed = 0
    for bench_code, bench_name, holdings_query in BENCHMARKS:
        try:
            # Process this benchmark
            result = get_bond_data(bench_code, bench_name, holdings_query, start_dt_str, end_dt_str)
            if result:
                successful_benchmarks += 1
                # Add the record count to the total
                total_records = len(pd.read_excel(result))
                total_records_processed += total_records
        except Exception as e:
            print(f"Error processing benchmark {bench_name}: {e}")
    
    print(f"Completed processing. Successfully processed {successful_benchmarks} of {len(BENCHMARKS)} benchmarks.")
    print(f"Total records processed: {total_records_processed}")

def main():
    """Main function to run the process"""
    try:
        print(f"Starting benchmark processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        run_benchmarks()
        print(f"Benchmark processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Fatal error in main process: {e}")

if __name__ == "__main__":
    main()