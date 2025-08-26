#!/usr/bin/env python3
"""
Command-line tool for setting up API keys for BMC Benchmark
"""

import sys
import argparse
from src.scripts.config import get_config_manager

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BMC Benchmark API Key Configuration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_api_keys.py                              # Interactive setup
  python setup_api_keys.py --list                       # List current keys
  python setup_api_keys.py --test gemini                # Test Gemini API key
  python setup_api_keys.py --remove openai              # Remove OpenAI key
  python setup_api_keys.py --set-model gemini gemini-2.5-flash  # Set Gemini model
  python setup_api_keys.py --list-models gemini         # List Gemini models
        """
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List current API key configuration'
    )
    
    parser.add_argument(
        '--test', '-t',
        metavar='PROVIDER',
        choices=['gemini', 'openai'],
        help='Test API key for specified provider'
    )
    
    parser.add_argument(
        '--remove', '-r',
        metavar='PROVIDER',
        choices=['gemini', 'openai'],
        help='Remove API key for specified provider'
    )
    
    parser.add_argument(
        '--set-default',
        metavar='PROVIDER',
        choices=['gemini', 'openai'],
        help='Set default LLM provider'
    )
    
    parser.add_argument(
        '--set-model',
        nargs=2,
        metavar=('PROVIDER', 'MODEL'),
        help='Set model for provider (e.g., --set-model gemini gemini-2.5-flash)'
    )
    
    parser.add_argument(
        '--list-models',
        metavar='PROVIDER',
        choices=['gemini', 'openai'],
        help='List available models for provider'
    )
    
    args = parser.parse_args()
    
    config_manager = get_config_manager()
    
    if args.list:
        config_manager.list_stored_keys()
    elif args.test:
        api_key = config_manager.get_api_key(args.test)
        if api_key:
            config_manager._test_api_key(args.test, api_key)
        else:
            print(f"❌ No API key found for {args.test}")
    elif args.remove:
        config_manager.remove_api_key(args.remove)
    elif args.set_default:
        config_manager.set_default_provider(args.set_default)
    elif args.set_model:
        provider, model = args.set_model
        if provider not in ['gemini', 'openai']:
            print(f"❌ Invalid provider: {provider}. Choose 'gemini' or 'openai'")
        else:
            config_manager.set_model_for_provider(provider, model)
    elif args.list_models:
        config_manager.list_available_models(args.list_models)
    else:
        # Interactive setup
        config_manager.setup_interactive()

if __name__ == "__main__":
    main()
