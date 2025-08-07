import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
import json
import time
from urllib.parse import urljoin, parse_qs, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SeriesInfo:
    """Information about a FOF series"""
    series_code: str
    series_name: str
    table_code: str
    line_number: int
    data_type: str = ""  # e.g., "Computed", "Source"
    formula: str = ""  # e.g., "= + FA106000105 + FA116000105"
    components: List[str] = field(default_factory=list)
    derived_from: List[Dict[str, str]] = field(default_factory=list)
    used_in: List[Dict[str, str]] = field(default_factory=list)
    shown_on: List[str] = field(default_factory=list)

@dataclass
class TableInfo:
    """Information about a FOF table"""
    table_code: str
    table_name: str
    table_type: str  # e.g., "Transactions", "Levels", "Balance"
    series_list: List[SeriesInfo] = field(default_factory=list)

@dataclass
class FOFIdentity:
    """Represents a Flow of Funds accounting identity"""
    identity_name: str
    table_codes: List[str]
    left_side: List[str]  # Series codes on left side
    right_side: List[str]  # Series codes on right side
    identity_type: str  # balance_sheet, flow_stock, sector_balance, market_clearing
    description: str

class FOFWebExtractor:
    """Extracts formulas and identities from Federal Reserve Flow of Funds website"""
    
    def __init__(self, use_cache=True):
        self.base_url = "https://www.federalreserve.gov"
        self.fof_base = "/apps/fof/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.use_cache = use_cache
        self.cache = {}
        self.tables = {}
        self.series_info = {}
        self.formulas = {}
        self.identities = []
        
    def extract_all_tables(self) -> Dict[str, TableInfo]:
        """Extract list of all tables from FOFTables.aspx"""
        url = urljoin(self.base_url, self.fof_base + "FOFTables.aspx")
        logger.info(f"Extracting table list from {url}")
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all table links
            table_links = soup.find_all('a', href=re.compile(r'DisplayTable\.aspx\?t='))
            
            for link in table_links:
                href = link.get('href', '')
                table_code = self.extract_table_code_from_url(href)
                if table_code:
                    table_name = link.get_text(strip=True)
                    
                    # Determine table type from code prefix
                    table_type = self._get_table_type(table_code)
                    
                    self.tables[table_code] = TableInfo(
                        table_code=table_code,
                        table_name=table_name,
                        table_type=table_type
                    )
            
            logger.info(f"Found {len(self.tables)} tables")
            return self.tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return {}
    
    def extract_table_code_from_url(self, url: str) -> Optional[str]:
        """Extract table code from URL"""
        match = re.search(r't=([A-Z]\.\d+[a-z]?(?:\.[a-z])?)', url)
        if match:
            return match.group(1)
        
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        return params.get('t', [None])[0]
    
    def _get_table_type(self, table_code: str) -> str:
        """Determine table type from code"""
        if table_code.startswith('F.'):
            return 'Transactions'
        elif table_code.startswith('L.'):
            return 'Levels'
        elif table_code.startswith('B.'):
            return 'Balance Sheet'
        elif table_code.startswith('R.'):
            return 'Reconciliation'
        elif table_code.startswith('D.'):
            return 'Debt'
        elif table_code.startswith('S.'):
            return 'Integrated Accounts'
        else:
            return 'Other'
    
    def extract_series_from_table(self, table_code: str) -> List[SeriesInfo]:
        """Extract all series from a specific table"""
        url = urljoin(self.base_url, f"{self.fof_base}DisplayTable.aspx?t={table_code}")
        logger.info(f"Extracting series from table {table_code}")
        
        series_list = []
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table with series data
            table = soup.find('table', id='gvwTableInfo')
            if not table:
                logger.warning(f"No series table found for {table_code}")
                return series_list
            
            # Extract series from table rows
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 3:
                    line_number = int(cells[0].get_text(strip=True))
                    
                    # Find series link
                    series_link = cells[1].find('a')
                    if series_link:
                        series_code = self.extract_series_code_from_cell(cells[1])
                        series_name = cells[2].get_text(strip=True)
                        
                        if series_code:
                            series = SeriesInfo(
                                series_code=series_code.replace('.Q', ''),  # Remove frequency suffix
                                series_name=series_name,
                                table_code=table_code,
                                line_number=line_number
                            )
                            series_list.append(series)
                            self.series_info[series_code] = series
            
            logger.info(f"Found {len(series_list)} series in table {table_code}")
            
        except Exception as e:
            logger.error(f"Error extracting series from table {table_code}: {e}")
        
        return series_list
    
    def extract_series_code_from_cell(self, cell) -> Optional[str]:
        """Extract series code from table cell"""
        # Try to find series code in link text
        link = cell.find('a')
        if link:
            text = link.get_text(strip=True)
            # Match FOF series code pattern
            match = re.match(r'([A-Z]{2}\d{9})', text)
            if match:
                return match.group(1)
        return None
    
    def extract_series_formula(self, series_code: str, table_code: str) -> Optional[SeriesInfo]:
        """Extract formula information for a specific series"""
        url = urljoin(self.base_url, 
                     f"{self.fof_base}SeriesAnalyzer.aspx?s={series_code}&t={table_code}&suf=Q")
        
        logger.info(f"Extracting formula for series {series_code}")
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get existing series info or create new
            series = self.series_info.get(series_code, SeriesInfo(
                series_code=series_code,
                series_name="",
                table_code=table_code,
                line_number=0
            ))
            
            # Extract series name
            name_elem = soup.find('span', id='lblSeriesDecription')
            if name_elem:
                series.series_name = name_elem.get_text(strip=True)
            
            # Extract data type (Computed, Source, etc.)
            data_type_elem = soup.find('span', id='lblDataType')
            if data_type_elem:
                series.data_type = data_type_elem.get_text(strip=True)
            
            # Extract formula
            formula_elem = soup.find('span', id='lblDataSource')
            if formula_elem:
                series.formula = formula_elem.get_text(strip=True)
                # Extract components from formula
                series.components = re.findall(r'[A-Z]{2}\d{9}', series.formula)
            
            # Extract "Derived from" information
            derived_table = soup.find('table', id='gvwSeriesDerived')
            if derived_table:
                rows = derived_table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        # Extract component code and operator
                        component_text = cells[0].get_text(strip=True)
                        operator_match = re.match(r'([+-])\s*([A-Z]{2}\d{9})', component_text)
                        if operator_match:
                            operator = operator_match.group(1)
                            component_code = operator_match.group(2)
                            component_name = cells[1].get_text(strip=True)
                            
                            series.derived_from.append({
                                'operator': operator,
                                'code': component_code,
                                'name': component_name
                            })
            
            # Extract "Used in" information
            used_table = soup.find('table', id='gvwSeriesUsed')
            if used_table:
                rows = used_table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        component_text = cells[0].get_text(strip=True)
                        operator_match = re.match(r'([+-])\s*([A-Z]{2}\d{9})', component_text)
                        if operator_match:
                            operator = operator_match.group(1)
                            component_code = operator_match.group(2)
                            component_name = cells[1].get_text(strip=True)
                            
                            series.used_in.append({
                                'operator': operator,
                                'code': component_code,
                                'name': component_name
                            })
            
            # Extract "Shown on" information
            shown_elem = soup.find('span', id='lblShowOn')
            if shown_elem:
                # Extract table references
                shown_links = shown_elem.find_all('a')
                for link in shown_links:
                    series.shown_on.append(link.get_text(strip=True))
            
            self.series_info[series_code] = series
            
            # Store as formula if it's computed
            if series.data_type == "Computed" and series.components:
                self.formulas[series_code] = series
            
            return series
            
        except Exception as e:
            logger.error(f"Error extracting formula for {series_code}: {e}")
            return None
    
    def find_identities_from_formulas(self):
        """Analyze formulas to find accounting identities"""
        # 1. Find balance sheet identities (Assets = Liabilities + Net Worth)
        self._find_balance_sheet_identities()
        
        # 2. Find flow-stock reconciliation identities
        self._find_flow_stock_identities()
        
        # 3. Find sector balance identities
        self._find_sector_balance_identities()
        
        # 4. Find market clearing identities
        self._find_market_clearing_identities()
        
        logger.info(f"Found {len(self.identities)} identities")
    
    def _find_balance_sheet_identities(self):
        """Find balance sheet identities (Assets = Liabilities + Net Worth)"""
        # Look for net worth series (typically ending in 090005)
        net_worth_series = {code: info for code, info in self.series_info.items() 
                           if code.endswith('090005')}
        
        for nw_code, nw_info in net_worth_series.items():
            # Extract sector code
            sector_code = nw_code[2:4]
            prefix = nw_code[:2]
            
            # Look for corresponding assets and liabilities
            assets_code = f"{prefix}{sector_code}000005"
            liabilities_code = f"{prefix}{sector_code}190005"
            
            # Check if this forms a valid identity
            if (assets_code in self.series_info and 
                liabilities_code in self.series_info):
                
                # Verify the formula
                if nw_info.formula and liabilities_code in nw_info.formula:
                    self.identities.append(FOFIdentity(
                        identity_name=f"Balance Sheet - {nw_info.series_name}",
                        table_codes=[nw_info.table_code],
                        left_side=[assets_code],
                        right_side=[liabilities_code, nw_code],
                        identity_type="balance_sheet",
                        description=f"Assets = Liabilities + Net Worth for {nw_info.series_name}"
                    ))
    
    def _find_flow_stock_identities(self):
        """Find flow-stock reconciliation identities"""
        # Group series by sector and instrument
        level_series = {code: info for code, info in self.series_info.items() 
                       if code.startswith('FL')}
        flow_series = {code: info for code, info in self.series_info.items() 
                      if code.startswith('FU') or code.startswith('FA')}
        
        for level_code, level_info in level_series.items():
            # Find corresponding flow series
            flow_code = 'FU' + level_code[2:]
            if flow_code in flow_series:
                self.identities.append(FOFIdentity(
                    identity_name=f"Flow-Stock - {level_info.series_name}",
                    table_codes=[level_info.table_code],
                    left_side=[f"Δ{level_code}"],
                    right_side=[flow_code, f"FR{level_code[2:]}", f"FO{level_code[2:]}"],
                    identity_type="flow_stock",
                    description="Change in Level = Flow + Revaluations + Other Changes"
                ))
    
    def _find_sector_balance_identities(self):
        """Find sector balance identities (S - I = Net Lending)"""
        # Look for net lending/borrowing series (typically ending in 005005)
        net_lending_series = {code: info for code, info in self.series_info.items() 
                             if code.endswith('005005')}
        
        for nl_code, nl_info in net_lending_series.items():
            sector_code = nl_code[2:4]
            prefix = nl_code[:2]
            
            # Look for saving and investment series
            saving_code = f"{prefix}{sector_code}006005"
            investment_code = f"{prefix}{sector_code}007005"
            
            if saving_code in self.series_info:
                self.identities.append(FOFIdentity(
                    identity_name=f"Sector Balance - {nl_info.series_name}",
                    table_codes=[nl_info.table_code],
                    left_side=[saving_code],
                    right_side=[investment_code, nl_code],
                    identity_type="sector_balance",
                    description="Saving - Investment = Net Lending/Borrowing"
                ))
    
    def _find_market_clearing_identities(self):
        """Find market clearing identities"""
        # Group series by instrument code
        instruments = {}
        for code, info in self.series_info.items():
            if len(code) >= 11:
                instrument = code[4:]
                if instrument not in instruments:
                    instruments[instrument] = []
                instruments[instrument].append((code, info))
        
        # For each instrument, check if total assets = total liabilities
        for instrument, series_list in instruments.items():
            if len(series_list) >= 2:
                # Separate into different sectors
                sectors = {}
                for code, info in series_list:
                    sector = code[2:4]
                    if sector not in ['89', '90', '91', '92', '93']:  # Exclude aggregate sectors
                        if sector not in sectors:
                            sectors[sector] = []
                        sectors[sector].append((code, info))
                
                if len(sectors) >= 2:
                    self.identities.append(FOFIdentity(
                        identity_name=f"Market Clearing - Instrument {instrument}",
                        table_codes=list(set(info.table_code for _, info in series_list)),
                        left_side=[code for code, _ in series_list if code.startswith('FL')],
                        right_side=[code for code, _ in series_list if code.startswith('FL')],
                        identity_type="market_clearing",
                        description="Total Assets = Total Liabilities across all sectors"
                    ))
    
    def extract_all_data(self, max_tables=None, max_series_per_table=None):
        """Extract all tables, series, and formulas"""
        # Step 1: Get all tables
        self.extract_all_tables()
        
        # Step 2: For each table, get all series
        table_count = 0
        for table_code, table_info in self.tables.items():
            if max_tables and table_count >= max_tables:
                break
                
            series_list = self.extract_series_from_table(table_code)
            table_info.series_list = series_list
            
            # Step 3: For each series, get formula details
            series_count = 0
            for series in series_list:
                if max_series_per_table and series_count >= max_series_per_table:
                    break
                    
                self.extract_series_formula(series.series_code, table_code)
                series_count += 1
                
                # Be respectful with rate limiting
                time.sleep(0.5)
            
            table_count += 1
            logger.info(f"Processed {table_count}/{len(self.tables)} tables")
        
        # Step 4: Find identities from formulas
        self.find_identities_from_formulas()
    
    def export_to_json(self, filename='fof_formulas_extracted.json'):
        """Export all extracted data to JSON"""
        # Convert dataclasses to dictionaries
        tables_dict = {code: asdict(table) for code, table in self.tables.items()}
        series_dict = {code: asdict(series) for code, series in self.series_info.items()}
        formulas_dict = {code: asdict(formula) for code, formula in self.formulas.items()}
        identities_dict = [asdict(identity) for identity in self.identities]
        
        # Build sector mapping
        sectors = {}
        for code, series in self.series_info.items():
            if len(code) >= 4:
                sector_code = code[2:4]
                sector_name = self._get_sector_name(sector_code)
                sectors[sector_code] = sector_name
        
        output = {
            'metadata': {
                'source': 'Federal Reserve Flow of Funds',
                'extraction_date': pd.Timestamp.now().isoformat(),
                'total_tables': len(self.tables),
                'total_series': len(self.series_info),
                'total_formulas': len(self.formulas),
                'total_identities': len(self.identities)
            },
            'tables': tables_dict,
            'series': series_dict,
            'formulas': formulas_dict,
            'identities': identities_dict,
            'sectors': sectors
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Exported data to {filename}")
        return output
    
    def _get_sector_name(self, sector_code: str) -> str:
        """Get sector name from code"""
        sectors = {
            '10': 'Households and nonprofit organizations',
            '11': 'Nonfinancial corporate business',
            '12': 'Nonfinancial noncorporate business',
            '13': 'Federal government',
            '14': 'State and local governments',
            '15': 'Financial business',
            '16': 'Private financial business',
            '17': 'Government financial business',
            '21': 'State and local governments',
            '26': 'Rest of the world',
            '31': 'Monetary authority',
            '36': 'Private depository institutions',
            '38': 'Domestic nonfinancial sectors',
            '40': 'Commercial banking',
            '41': 'Savings institutions',
            '42': 'Credit unions',
            '50': 'Life insurance companies',
            '51': 'Property-casualty insurance companies',
            '54': 'Private pension funds',
            '55': 'State and local govt retirement funds',
            '61': 'Money market funds',
            '62': 'Mutual funds',
            '63': 'Closed-end funds',
            '64': 'Exchange-traded funds',
            '66': 'Government-sponsored enterprises',
            '67': 'Agency-backed mortgage pools',
            '70': 'Asset-backed securities issuers',
            '71': 'Finance companies',
            '73': 'Real estate investment trusts',
            '74': 'Security brokers and dealers',
            '75': 'Holding companies',
            '76': 'Funding corporations',
            '79': 'Financial sectors n.e.c.',
            '89': 'All sectors',
            '90': 'Domestic nonfinancial sectors',
            '91': 'Private domestic nonfinancial',
            '92': 'Rest of the world',
            '93': 'All domestic sectors',
            '94': 'Private domestic sectors',
            '95': 'Government sectors',
            '96': 'Domestic financial sectors'
        }
        return sectors.get(sector_code, f'Sector {sector_code}')
    
    def generate_summary_report(self):
        """Generate a summary report of findings"""
        report = []
        report.append("Federal Reserve Flow of Funds Formula Analysis")
        report.append("=" * 50)
        report.append(f"\nTotal Tables: {len(self.tables)}")
        report.append(f"Total Series: {len(self.series_info)}")
        report.append(f"Total Formulas: {len(self.formulas)}")
        report.append(f"Total Identities: {len(self.identities)}")
        
        # Table type breakdown
        table_types = {}
        for table in self.tables.values():
            table_types[table.table_type] = table_types.get(table.table_type, 0) + 1
        
        report.append("\nTables by Type:")
        for ttype, count in sorted(table_types.items()):
            report.append(f"  {ttype}: {count}")
        
        # Identity type breakdown
        identity_types = {}
        for identity in self.identities:
            identity_types[identity.identity_type] = identity_types.get(identity.identity_type, 0) + 1
        
        report.append("\nIdentities by Type:")
        for itype, count in sorted(identity_types.items()):
            report.append(f"  {itype}: {count}")
        
        # Sample formulas
        report.append("\nSample Formulas:")
        for i, (code, formula) in enumerate(list(self.formulas.items())[:5]):
            report.append(f"\n{i+1}. {code}: {formula.series_name}")
            report.append(f"   Formula: {formula.formula}")
            report.append(f"   Components: {', '.join(formula.components)}")
        
        # Sample identities
        report.append("\n\nSample Identities:")
        for i, identity in enumerate(self.identities[:5]):
            report.append(f"\n{i+1}. {identity.identity_name}")
            report.append(f"   Type: {identity.identity_type}")
            report.append(f"   Description: {identity.description}")
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = FOFWebExtractor()
    
    # Extract data with limits for testing
    # In production, remove these limits to get all data
    # extractor.extract_all_data(max_tables=5, max_series_per_table=10)
    extractor.extract_all_data()
    
    # Export to JSON
    extractor.export_to_json()
    
    # Generate and print summary report
    report = extractor.generate_summary_report()
    print(report)
    
    # Example: Get specific series formula
    series_info = extractor.extract_series_formula("FA386000105", "F.100")
    if series_info:
        print(f"\nDetailed info for {series_info.series_code}:")
        print(f"Name: {series_info.series_name}")
        print(f"Formula: {series_info.formula}")
        print(f"Components: {series_info.components}")
