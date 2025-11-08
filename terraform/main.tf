# Kubera Infrastructure as Code
# Terraform configuration for Railway deployment

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    railway = {
      source  = "terraform-community-providers/railway"
      version = "~> 0.3"
    }
  }

  # Backend for state management (optional - use S3, Terraform Cloud, etc.)
  # backend "s3" {
  #   bucket = "kubera-terraform-state"
  #   key    = "production/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

# ============================================
# Variables
# ============================================

variable "railway_token" {
  description = "Railway API token"
  type        = string
  sensitive   = true
}

variable "openrouter_api_key" {
  description = "OpenRouter API key for AI models"
  type        = string
  sensitive   = true
}

variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API key for market data"
  type        = string
  sensitive   = true
}

variable "telegram_bot_token" {
  description = "Telegram Bot API token"
  type        = string
  sensitive   = true
}

variable "environment" {
  description = "Environment (production, staging, development)"
  type        = string
  default     = "production"
}

variable "github_repo" {
  description = "GitHub repository URL"
  type        = string
  default     = "https://github.com/YOUR_USERNAME/kubera"
}

# ============================================
# Railway Provider
# ============================================

provider "railway" {
  token = var.railway_token
}

# ============================================
# Railway Project
# ============================================

resource "railway_project" "kubera" {
  name        = "kubera-${var.environment}"
  description = "Kubera Autonomous Trading System - ${var.environment}"
}

# ============================================
# Railway Service
# ============================================

resource "railway_service" "bot" {
  project_id  = railway_project.kubera.id
  name        = "kubera-bot"
  source_repo = var.github_repo
  source_branch = var.environment == "production" ? "main" : "develop"

  # Build configuration
  build_command = ""  # Uses Dockerfile
  start_command = "python -u src/bot/telegram_bot.py"

  # Environment variables
  variables = {
    ENVIRONMENT            = var.environment
    LOG_LEVEL             = var.environment == "production" ? "INFO" : "DEBUG"
    OPENROUTER_API_KEY    = var.openrouter_api_key
    ALPHA_VANTAGE_API_KEY = var.alpha_vantage_api_key
    TELEGRAM_BOT_TOKEN    = var.telegram_bot_token
  }

  # Service configuration
  restart_policy_type = "on_failure"
  restart_policy_max_retries = 10
}

# ============================================
# Outputs
# ============================================

output "project_id" {
  description = "Railway project ID"
  value       = railway_project.kubera.id
}

output "service_id" {
  description = "Railway service ID"
  value       = railway_service.bot.id
}

output "service_url" {
  description = "Railway service URL"
  value       = "https://kubera-${var.environment}.railway.app"
}

