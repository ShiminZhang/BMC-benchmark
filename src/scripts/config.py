#!/usr/bin/env python3
"""
Configuration management for API keys and settings
Provides secure storage and loading of API keys with multiple fallback options
"""

import os
import json
import getpass
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import cryptography for encryption, fall back to base64 if not available
try:
    from cryptography.fernet import Fernet
    import base64
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    print("Note: cryptography not installed. API keys will be stored with basic encoding only.")

class ConfigManager:
    """Manages API keys and configuration with multiple storage options"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            config_dir: Custom config directory path
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to user's home directory
            self.config_dir = Path.home() / ".bmc_benchmark"
        
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.encrypted_file = self.config_dir / "keys.enc"
        self.key_file = self.config_dir / ".keyfile"
        
        # Load configuration
        self._config = self._load_config()
    
    def _generate_key(self) -> bytes:
        """Generate encryption key"""
        return Fernet.generate_key()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get existing encryption key or create new one"""
        if self.key_file.exists():
            try:
                with open(self.key_file, 'rb') as f:
                    return f.read()
            except Exception:
                pass
        
        # Create new key
        key = self._generate_key()
        try:
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Make key file readable only by owner
            os.chmod(self.key_file, 0o600)
        except Exception as e:
            print(f"Warning: Could not save encryption key: {e}")
        
        return key
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not ENCRYPTION_AVAILABLE:
            # Simple base64 encoding as fallback (not secure, but better than plain text)
            import base64
            return base64.b64encode(data.encode()).decode()
        
        try:
            key = self._get_or_create_encryption_key()
            f = Fernet(key)
            encrypted = f.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            print(f"Warning: Encryption failed: {e}")
            # Fallback to base64 encoding
            import base64
            return base64.b64encode(data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not ENCRYPTION_AVAILABLE:
            # Simple base64 decoding as fallback
            try:
                import base64
                return base64.b64decode(encrypted_data.encode()).decode()
            except Exception:
                return encrypted_data  # Return as-is if decoding fails
        
        try:
            key = self._get_or_create_encryption_key()
            f = Fernet(key)
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = f.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            # Try base64 decoding as fallback
            try:
                import base64
                return base64.b64decode(encrypted_data.encode()).decode()
            except Exception:
                return encrypted_data  # Return as-is if all decryption fails
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
        
        return {
            "api_keys": {},
            "default_provider": "gemini",
            "models": {
                "gemini": "gemini-2.5-flash",
                "openai": "gpt-4"
            }
        }
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            # Make config file readable only by owner
            os.chmod(self.config_file, 0o600)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def set_api_key(self, provider: str, api_key: str, encrypt: bool = True):
        """Store API key securely
        
        Args:
            provider: Provider name (gemini, openai)
            api_key: The API key to store
            encrypt: Whether to encrypt the key (default: True)
        """
        if encrypt:
            encrypted_key = self._encrypt_data(api_key)
            self._config["api_keys"][provider] = {
                "key": encrypted_key,
                "encrypted": True
            }
        else:
            self._config["api_keys"][provider] = {
                "key": api_key,
                "encrypted": False
            }
        
        self._save_config()
        print(f"✅ API key for {provider} saved successfully")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key with multiple fallback options
        
        Args:
            provider: Provider name (gemini, openai)
            
        Returns:
            API key if found, None otherwise
        """
        # Priority order:
        # 1. Environment variable
        # 2. Stored encrypted key
        # 3. Stored plain key
        
        # Check environment variables first
        env_vars = {
            "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "openai": ["OPENAI_API_KEY"]
        }
        
        for env_var in env_vars.get(provider, []):
            key = os.getenv(env_var)
            if key:
                return key
        
        # Check stored keys
        stored_key_info = self._config["api_keys"].get(provider)
        if stored_key_info:
            key = stored_key_info["key"]
            if stored_key_info.get("encrypted", False):
                return self._decrypt_data(key)
            else:
                return key
        
        return None
    
    def remove_api_key(self, provider: str):
        """Remove stored API key"""
        if provider in self._config["api_keys"]:
            del self._config["api_keys"][provider]
            self._save_config()
            print(f"✅ API key for {provider} removed")
        else:
            print(f"⚠️  No stored API key found for {provider}")
    
    def list_stored_keys(self):
        """List all stored API key providers"""
        stored = list(self._config["api_keys"].keys())
        env_keys = []
        
        # Check environment variables
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            env_keys.append("gemini")
        if os.getenv("OPENAI_API_KEY"):
            env_keys.append("openai")
        
        print("🔑 API Key Status:")
        print("-" * 20)
        
        for provider in ["gemini", "openai"]:
            status = []
            if provider in stored:
                key_info = self._config["api_keys"][provider]
                if key_info.get("encrypted", False):
                    status.append("stored (encrypted)")
                else:
                    status.append("stored (plain)")
            
            if provider in env_keys:
                status.append("environment")
            
            if status:
                print(f"{provider}: {', '.join(status)}")
            else:
                print(f"{provider}: not configured")
    
    def setup_interactive(self):
        """Interactive setup for API keys"""
        print("🔧 BMC Benchmark API Key Setup")
        print("=" * 40)
        
        providers = {
            "1": ("gemini", "Google Gemini API Key"),
            "2": ("openai", "OpenAI API Key")
        }
        
        while True:
            print("\nSelect option:")
            for key, (provider, name) in providers.items():
                current_key = self.get_api_key(provider)
                status = "✅ configured" if current_key else "❌ not configured"
                current_model = self.get_model_for_provider(provider)
                print(f"{key}. {name} ({status}, model: {current_model})")
            print("3. List current configuration")
            print("4. Remove API key")
            print("5. Configure models")
            print("6. Set default provider")
            print("7. Exit")
            
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice in providers:
                provider, name = providers[choice]
                self._setup_provider_key(provider, name)
            elif choice == "3":
                self.list_stored_keys()
            elif choice == "4":
                self._remove_key_interactive()
            elif choice == "5":
                self._configure_models_interactive()
            elif choice == "6":
                self._set_default_provider_interactive()
            elif choice == "7":
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _setup_provider_key(self, provider: str, name: str):
        """Setup API key for specific provider"""
        print(f"\n🔑 Setting up {name}")
        print("-" * 30)
        
        current_key = self.get_api_key(provider)
        if current_key:
            print(f"Current key: {current_key[:10]}...{current_key[-4:]}")
            if input("Replace existing key? (y/N): ").lower() != 'y':
                return
        
        # Get API key
        api_key = getpass.getpass(f"Enter your {name}: ").strip()
        if not api_key:
            print("No API key entered. Skipping.")
            return
        
        # Ask about encryption
        encrypt = input("Encrypt the key? (Y/n): ").lower() != 'n'
        
        # Store the key
        self.set_api_key(provider, api_key, encrypt)
        
        # Test the key
        if input("Test the API key? (Y/n): ").lower() != 'n':
            self._test_api_key(provider, api_key)
    
    def _remove_key_interactive(self):
        """Interactive API key removal"""
        stored_keys = list(self._config["api_keys"].keys())
        if not stored_keys:
            print("No stored API keys to remove.")
            return
        
        print("\nStored API keys:")
        for i, provider in enumerate(stored_keys, 1):
            print(f"{i}. {provider}")
        
        try:
            choice = int(input("Enter number to remove (0 to cancel): "))
            if 0 < choice <= len(stored_keys):
                provider = stored_keys[choice - 1]
                self.remove_api_key(provider)
        except ValueError:
            print("Invalid choice.")
    
    def _configure_models_interactive(self):
        """Interactive model configuration"""
        print("\n🤖 MODEL CONFIGURATION")
        print("=" * 30)
        
        providers = ["gemini", "openai"]
        
        print("Select provider to configure model:")
        for i, provider in enumerate(providers, 1):
            current_model = self.get_model_for_provider(provider)
            print(f"{i}. {provider} (current: {current_model})")
        
        try:
            choice = int(input("Enter choice (1-2): "))
            if 1 <= choice <= len(providers):
                provider = providers[choice - 1]
                self._setup_model_for_provider(provider)
        except ValueError:
            print("Invalid choice.")
    
    def _setup_model_for_provider(self, provider: str):
        """Setup model for specific provider"""
        print(f"\n🔧 Configuring {provider} model")
        print("-" * 25)
        
        current_model = self.get_model_for_provider(provider)
        print(f"Current model: {current_model}")
        
        # List available models
        self.list_available_models(provider)
        
        # Get new model
        new_model = input(f"\nEnter new model name (or press Enter to keep {current_model}): ").strip()
        if new_model:
            self.set_model_for_provider(provider, new_model)
            
            # Test the new model if API key is available
            api_key = self.get_api_key(provider)
            if api_key and input("Test the new model? (Y/n): ").lower() != 'n':
                self._test_model(provider, api_key, new_model)
    
    def _set_default_provider_interactive(self):
        """Interactive default provider setting"""
        print("\n🎯 SET DEFAULT PROVIDER")
        print("=" * 25)
        
        providers = ["gemini", "openai"]
        current_default = self.get_default_provider()
        
        print(f"Current default: {current_default}")
        print("\nSelect new default provider:")
        for i, provider in enumerate(providers, 1):
            marker = "👈 current" if provider == current_default else ""
            print(f"{i}. {provider} {marker}")
        
        try:
            choice = int(input("Enter choice (1-2): "))
            if 1 <= choice <= len(providers):
                new_provider = providers[choice - 1]
                if new_provider != current_default:
                    self.set_default_provider(new_provider)
                else:
                    print(f"{new_provider} is already the default provider")
        except ValueError:
            print("Invalid choice.")
    
    def _test_model(self, provider: str, api_key: str, model: str):
        """Test specific model"""
        print(f"Testing {provider} model '{model}'...")
        
        try:
            if provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                test_model = genai.GenerativeModel(model)
                response = test_model.generate_content("Say 'Hello' if you can see this.")
                print(f"✅ {provider} model '{model}' is working!")
                
            elif provider == "openai":
                import openai
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say 'Hello' if you can see this."}],
                    max_tokens=10
                )
                print(f"✅ {provider} model '{model}' is working!")
                
        except Exception as e:
            print(f"❌ {provider} model '{model}' test failed: {e}")
            print("You may want to choose a different model.")
    
    def _test_api_key(self, provider: str, api_key: str):
        """Test API key validity"""
        print(f"Testing {provider} API key...")
        
        try:
            if provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content("Say 'Hello' if you can see this.")
                print(response)
                print(f"✅ {provider} API key is working!")
                
            elif provider == "openai":
                import openai
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say 'Hello' if you can see this."}],
                    max_tokens=10
                )
                print(f"✅ {provider} API key is working!")
                
        except Exception as e:
            print(f"❌ {provider} API key test failed: {e}")
    
    def get_default_provider(self) -> str:
        """Get default LLM provider"""
        return self._config.get("default_provider", "gemini")
    
    def set_default_provider(self, provider: str):
        """Set default LLM provider"""
        self._config["default_provider"] = provider
        self._save_config()
        print(f"✅ Default provider set to {provider}")
    
    def set_model_for_provider(self, provider: str, model: str):
        """Set model for specific provider"""
        if "models" not in self._config:
            self._config["models"] = {}
        self._config["models"][provider] = model
        self._save_config()
        print(f"✅ Model for {provider} set to {model}")
    
    def list_available_models(self, provider: str):
        """List available models for provider"""
        if provider == "gemini":
            try:
                import google.generativeai as genai
                
                # Try to use stored API key
                api_key = self.get_api_key(provider)
                if api_key:
                    genai.configure(api_key=api_key)
                
                models = genai.list_models()
                print(f"\n🤖 Available Gemini models:")
                print("-" * 30)
                for model in models:
                    if 'generateContent' in model.supported_generation_methods:
                        print(f"  • {model.name.replace('models/', '')}")
            except Exception as e:
                print(f"❌ Could not list Gemini models: {e}")
                print("Common Gemini models:")
                print("  • gemini-2.5-flash")
                print("  • gemini-1.5-pro")
                print("  • gemini-1.5-flash")
        
        elif provider == "openai":
            print(f"\n🤖 Common OpenAI models:")
            print("-" * 30)
            print("  • gpt-4")
            print("  • gpt-4-turbo")
            print("  • gpt-3.5-turbo")
            print("  • gpt-4o")
            print("  • gpt-4o-mini")
    
    def get_model_for_provider(self, provider: str) -> str:
        """Get default model for provider"""
        return self._config.get("models", {}).get(provider, "gemini-2.5-flash" if provider == "gemini" else "gpt-4")


# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_api_key(provider: str) -> Optional[str]:
    """Convenience function to get API key"""
    return get_config_manager().get_api_key(provider)

def setup_config():
    """Convenience function to run interactive setup"""
    get_config_manager().setup_interactive()


if __name__ == "__main__":
    # Run interactive setup if called directly
    setup_config()
