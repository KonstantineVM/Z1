#!/usr/bin/env python
"""
Cache Manager CLI Tool
Manage cache for Economic Time Series Analysis Framework
"""

import click
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from tabulate import tabulate
import shutil
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CachedFedDataLoader, ExternalDataLoader


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_size(size_bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_directory_size(directory):
    """Get total size of directory"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total


@click.group()
@click.option('--config', '-c', default='config/config.yaml', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Economic Time Series Analysis Cache Manager"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)


@cli.command()
@click.pass_context
def status(ctx):
    """Show cache status and statistics"""
    config = ctx.obj['config']
    data_config = config.get('data', {})
    
    click.echo("=== Cache Status ===\n")
    
    # Federal Reserve data cache
    click.echo("Federal Reserve Data Cache:")
    fed_loader = CachedFedDataLoader(
        cache_directory=data_config.get('cache_directory', './data/cache'),
        cache_expiry_days=data_config.get('cache_expiry_days', 7)
    )
    
    cache_info = fed_loader.get_cache_info()
    
    if cache_info:
        # Prepare table data
        table_data = []
        total_size = 0
        valid_count = 0
        expired_count = 0
        
        for source, info in cache_info.items():
            cached_at = datetime.fromisoformat(info['cached_at'])
            age_days = (datetime.now() - cached_at).days
            status = "✓ Valid" if info['valid'] else "✗ Expired"
            
            if info['valid']:
                valid_count += 1
            else:
                expired_count += 1
                
            size = info.get('size_bytes', 0)
            total_size += size
            
            shape = f"{info['shape'][0]} × {info['shape'][1]}"
            
            table_data.append([
                source,
                cached_at.strftime("%Y-%m-%d %H:%M"),
                f"{age_days} days",
                shape,
                format_size(size),
                status
            ])
        
        # Display table
        headers = ['Source', 'Cached At', 'Age', 'Shape', 'Size', 'Status']
        click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
        click.echo(f"\nTotal: {len(cache_info)} sources, {valid_count} valid, {expired_count} expired")
        click.echo(f"Total cache size: {format_size(total_size)}")
    else:
        click.echo("No cached data found")
    
    # External data cache
    click.echo("\n\nExternal Data Cache:")
    external_loader = ExternalDataLoader(
        cache_directory=data_config.get('external_data', {}).get('cache_directory', './data/cache/external')
    )
    
    external_info = external_loader.get_cache_info()
    if external_info:
        external_data = []
        for source, info in external_info.items():
            cached_at = datetime.fromisoformat(info['cached_at'])
            age_days = (datetime.now() - cached_at).days
            status = "✓ Valid" if info['valid'] else "✗ Expired"
            
            external_data.append([
                source,
                cached_at.strftime("%Y-%m-%d %H:%M"),
                f"{age_days} days",
                status
            ])
        
        click.echo(tabulate(external_data, 
                          headers=['Source', 'Cached At', 'Age', 'Status'],
                          tablefmt='grid'))
    else:
        click.echo("No external data cached")


@cli.command()
@click.option('--source', '-s', help='Specific source to clear')
@click.option('--type', '-t', type=click.Choice(['fed', 'external', 'all']), 
              default='all', help='Type of cache to clear')
@click.option('--force', '-f', is_flag=True, help='Force clear without confirmation')
@click.pass_context
def clear(ctx, source, type, force):
    """Clear cache for specified sources"""
    config = ctx.obj['config']
    data_config = config.get('data', {})
    
    # Confirmation
    if not force:
        if source:
            message = f"Clear cache for {source}?"
        else:
            message = f"Clear {type} cache?"
        
        if not click.confirm(message):
            click.echo("Cancelled")
            return
    
    # Clear Federal Reserve cache
    if type in ['fed', 'all']:
        fed_loader = CachedFedDataLoader(
            cache_directory=data_config.get('cache_directory', './data/cache')
        )
        
        if source and type == 'fed':
            fed_loader.clear_cache(source)
            click.echo(f"✓ Cleared Fed cache for {source}")
        elif type == 'fed' or (type == 'all' and not source):
            fed_loader.clear_cache()
            click.echo("✓ Cleared all Fed cache")
    
    # Clear external cache
    if type in ['external', 'all']:
        external_loader = ExternalDataLoader(
            cache_directory=data_config.get('external_data', {}).get('cache_directory', './data/cache/external')
        )
        
        if source and type == 'external':
            external_loader.clear_cache(source)
            click.echo(f"✓ Cleared external cache for {source}")
        elif type == 'external' or (type == 'all' and not source):
            external_loader.clear_cache()
            click.echo("✓ Cleared all external cache")
    
    click.echo("\nCache cleared successfully")


@cli.command()
@click.option('--sources', '-s', multiple=True, help='Specific sources to refresh')
@click.option('--type', '-t', type=click.Choice(['fed', 'external', 'all']), 
              default='all', help='Type of cache to refresh')
@click.option('--parallel', '-p', is_flag=True, help='Refresh in parallel')
@click.pass_context
def refresh(ctx, sources, type, parallel):
    """Refresh cache by downloading fresh data"""
    config = ctx.obj['config']
    data_config = config.get('data', {})
    
    click.echo("Refreshing cache...\n")
    
    # Refresh Federal Reserve data
    if type in ['fed', 'all']:
        fed_loader = CachedFedDataLoader(
            cache_directory=data_config.get('cache_directory', './data/cache'),
            start_year=data_config.get('fed_data', {}).get('start_year', 1960),
            end_year=data_config.get('fed_data', {}).get('end_year', 2024)
        )
        
        if sources:
            fed_sources = [s for s in sources if s in data_config.get('fed_data', {}).get('sources', [])]
        else:
            fed_sources = data_config.get('fed_data', {}).get('sources', [])
        
        if fed_sources:
            click.echo(f"Refreshing Fed sources: {', '.join(fed_sources)}")
            
            if parallel and len(fed_sources) > 1:
                with click.progressbar(length=len(fed_sources), 
                                     label='Downloading Fed data') as bar:
                    results = fed_loader.load_multiple_sources(
                        fed_sources, 
                        force_refresh=True,
                        n_jobs=-1
                    )
                    bar.update(len(fed_sources))
            else:
                for source in fed_sources:
                    click.echo(f"  Downloading {source}...", nl=False)
                    try:
                        fed_loader.load_source(source, force_refresh=True)
                        click.echo(" ✓")
                    except Exception as e:
                        click.echo(f" ✗ Error: {str(e)}")
    
    # Refresh external data
    if type in ['external', 'all']:
        external_loader = ExternalDataLoader(
            cache_directory=data_config.get('external_data', {}).get('cache_directory', './data/cache/external')
        )
        
        external_sources = []
        if sources:
            external_sources = [s for s in sources if s in ['sp500', 'gold', 'fred']]
        else:
            external_sources = ['sp500', 'gold', 'fred']
        
        if external_sources:
            click.echo(f"\nRefreshing external sources: {', '.join(external_sources)}")
            
            for source in external_sources:
                click.echo(f"  Downloading {source}...", nl=False)
                try:
                    if source == 'sp500':
                        external_loader.load_sp500(force_refresh=True)
                    elif source == 'gold':
                        external_loader.load_gold_prices(force_refresh=True)
                    elif source == 'fred':
                        fred_config = data_config.get('external_data', {}).get('fred', {})
                        series = fred_config.get('series', [])
                        for s in series:
                            external_loader.load_fred_series(s, force_refresh=True)
                    click.echo(" ✓")
                except Exception as e:
                    click.echo(f" ✗ Error: {str(e)}")
    
    click.echo("\nCache refresh complete")


@cli.command()
@click.option('--expiry-days', '-e', type=int, help='Set cache expiry in days')
@click.option('--show', '-s', is_flag=True, help='Show current configuration')
@click.pass_context
def config(ctx, expiry_days, show):
    """Configure cache settings"""
    config_data = ctx.obj['config']
    config_path = 'config/config.yaml'
    
    if show:
        click.echo("Current cache configuration:")
        click.echo(f"  Cache directory: {config_data.get('data', {}).get('cache_directory')}")
        click.echo(f"  Cache expiry: {config_data.get('data', {}).get('cache_expiry_days')} days")
        click.echo(f"  Force download: {config_data.get('data', {}).get('force_download')}")
        return
    
    if expiry_days is not None:
        # Update configuration
        config_data['data']['cache_expiry_days'] = expiry_days
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        click.echo(f"✓ Updated cache expiry to {expiry_days} days")


@cli.command()
@click.pass_context
def clean(ctx):
    """Clean expired cache entries"""
    config = ctx.obj['config']
    data_config = config.get('data', {})
    
    click.echo("Cleaning expired cache entries...\n")
    
    # Clean Federal Reserve cache
    fed_loader = CachedFedDataLoader(
        cache_directory=data_config.get('cache_directory', './data/cache'),
        cache_expiry_days=data_config.get('cache_expiry_days', 7)
    )
    
    cache_info = fed_loader.get_cache_info()
    cleaned_count = 0
    
    for source, info in cache_info.items():
        if not info['valid']:
            fed_loader.clear_cache(source)
            cleaned_count += 1
            click.echo(f"  Removed expired cache for {source}")
    
    if cleaned_count > 0:
        click.echo(f"\n✓ Cleaned {cleaned_count} expired entries")
    else:
        click.echo("No expired entries found")


@cli.command()
@click.argument('source')
@click.pass_context
def info(ctx, source):
    """Show detailed information about cached source"""
    config = ctx.obj['config']
    data_config = config.get('data', {})
    
    # Check Federal Reserve cache
    fed_loader = CachedFedDataLoader(
        cache_directory=data_config.get('cache_directory', './data/cache')
    )
    
    cache_info = fed_loader.get_cache_info()
    
    if source in cache_info:
        info = cache_info[source]
        click.echo(f"\n=== {source} Cache Information ===")
        click.echo(f"Cached at: {info['cached_at']}")
        click.echo(f"Valid: {'Yes' if info['valid'] else 'No'}")
        click.echo(f"Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
        click.echo(f"Size: {format_size(info.get('size_bytes', 0))}")
        click.echo(f"Start year: {info.get('start_year', 'N/A')}")
        click.echo(f"End year: {info.get('end_year', 'N/A')}")
        
        # Show sample data if available
        try:
            data = fed_loader._load_from_cache(source)
            click.echo(f"\nColumns ({len(data.columns)}):")
            for i, col in enumerate(data.columns[:10]):
                click.echo(f"  - {col}")
            if len(data.columns) > 10:
                click.echo(f"  ... and {len(data.columns) - 10} more")
                
            click.echo(f"\nData range: {data.index.min()} to {data.index.max()}")
        except Exception as e:
            click.echo(f"\nCould not load data: {str(e)}")
    else:
        click.echo(f"No cache found for source: {source}")


@cli.command()
@click.option('--output', '-o', default='cache_report.txt', help='Output file for report')
@click.pass_context
def report(ctx, output):
    """Generate detailed cache report"""
    config = ctx.obj['config']
    
    with open(output, 'w') as f:
        f.write("Economic Time Series Analysis - Cache Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write configuration
        f.write("Configuration:\n")
        f.write(f"  Cache directory: {config.get('data', {}).get('cache_directory')}\n")
        f.write(f"  Cache expiry: {config.get('data', {}).get('cache_expiry_days')} days\n\n")
        
        # Get cache statistics
        data_config = config.get('data', {})
        fed_loader = CachedFedDataLoader(
            cache_directory=data_config.get('cache_directory', './data/cache')
        )
        
        cache_info = fed_loader.get_cache_info()
        
        if cache_info:
            f.write("Cached Sources:\n")
            for source, info in cache_info.items():
                f.write(f"\n  {source}:\n")
                f.write(f"    Cached at: {info['cached_at']}\n")
                f.write(f"    Valid: {'Yes' if info['valid'] else 'No'}\n")
                f.write(f"    Shape: {info['shape']}\n")
                f.write(f"    Size: {format_size(info.get('size_bytes', 0))}\n")
        
        # Calculate summary statistics
        total_sources = len(cache_info)
        valid_sources = sum(1 for info in cache_info.values() if info['valid'])
        total_size = sum(info.get('size_bytes', 0) for info in cache_info.values())
        
        f.write(f"\nSummary:\n")
        f.write(f"  Total sources: {total_sources}\n")
        f.write(f"  Valid sources: {valid_sources}\n")
        f.write(f"  Expired sources: {total_sources - valid_sources}\n")
        f.write(f"  Total cache size: {format_size(total_size)}\n")
    
    click.echo(f"✓ Report saved to {output}")


if __name__ == '__main__':
    cli()