#!/bin/bash
# HyperOpt Strategy Documentation Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCS_DIR="docs"
BUILD_DIR="site"
DEPLOY_BRANCH="gh-pages"
REMOTE_URL="https://github.com/hyperopt-strat/platform.git"

echo -e "${BLUE}🚀 HyperOpt Strategy Documentation Deployment${NC}"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "mkdocs.yml" ]; then
    echo -e "${RED}❌ Error: mkdocs.yml not found. Please run from project root.${NC}"
    exit 1
fi

# Check if MkDocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo -e "${RED}❌ Error: MkDocs is not installed.${NC}"
    echo "Install with: pip install mkdocs-material"
    exit 1
fi

echo -e "${YELLOW}📋 Pre-deployment checks...${NC}"

# Check for required dependencies
REQUIRED_PACKAGES=(
    "mkdocs-material"
    "mkdocs-glightbox"
    "mkdocs-git-revision-date-localized-plugin"
    "mkdocs-git-committers-plugin"
    "mkdocs-minify-plugin"
    "mkdocstrings[python]"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "${package%[*}" &> /dev/null; then
        echo -e "${RED}❌ Missing package: $package${NC}"
        echo "Install with: pip install $package"
        exit 1
    fi
done

echo -e "${GREEN}✅ All required packages are installed${NC}"

# Validate configuration
echo -e "${YELLOW}🔍 Validating MkDocs configuration...${NC}"
if ! mkdocs config-validate; then
    echo -e "${RED}❌ Invalid MkDocs configuration${NC}"
    exit 1
fi

echo -e "${GREEN}✅ MkDocs configuration is valid${NC}"

# Build documentation
echo -e "${YELLOW}🔨 Building documentation...${NC}"
if ! mkdocs build --clean --strict; then
    echo -e "${RED}❌ Documentation build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Documentation built successfully${NC}"

# Check build output
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}❌ Build directory not found${NC}"
    exit 1
fi

# Calculate build size
BUILD_SIZE=$(du -sh "$BUILD_DIR" | cut -f1)
echo -e "${BLUE}📊 Build size: $BUILD_SIZE${NC}"

# Count generated files
FILE_COUNT=$(find "$BUILD_DIR" -type f | wc -l)
echo -e "${BLUE}📁 Generated files: $FILE_COUNT${NC}"

# Performance check
echo -e "${YELLOW}⚡ Running performance checks...${NC}"

# Check for large files
LARGE_FILES=$(find "$BUILD_DIR" -type f -size +1M)
if [ -n "$LARGE_FILES" ]; then
    echo -e "${YELLOW}⚠️  Large files detected (>1MB):${NC}"
    echo "$LARGE_FILES" | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "  - $file ($size)"
    done
fi

# Check for uncompressed images
UNCOMPRESSED_IMAGES=$(find "$BUILD_DIR" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | head -10)
if [ -n "$UNCOMPRESSED_IMAGES" ]; then
    echo -e "${YELLOW}⚠️  Consider optimizing images for better performance${NC}"
fi

# Deployment mode selection
echo ""
echo -e "${BLUE}🎯 Select deployment target:${NC}"
echo "1) GitHub Pages (production)"
echo "2) Local preview"
echo "3) Custom server"
echo "4) Exit"

read -p "Choose deployment option (1-4): " choice

case $choice in
    1)
        echo -e "${YELLOW}🌐 Deploying to GitHub Pages...${NC}"
        
        # Check for git
        if ! command -v git &> /dev/null; then
            echo -e "${RED}❌ Git is not installed${NC}"
            exit 1
        fi
        
        # Check if we're in a git repository
        if ! git rev-parse --git-dir > /dev/null 2>&1; then
            echo -e "${RED}❌ Not in a git repository${NC}"
            exit 1
        fi
        
        # Check for uncommitted changes
        if ! git diff-index --quiet HEAD --; then
            echo -e "${YELLOW}⚠️  You have uncommitted changes. Commit them first? (y/n)${NC}"
            read -r commit_choice
            if [[ $commit_choice =~ ^[Yy]$ ]]; then
                git add .
                git commit -m "Update documentation before deployment"
            fi
        fi
        
        # Deploy to GitHub Pages
        if ! mkdocs gh-deploy --clean --message "Deploy documentation {sha} via deploy-docs.sh"; then
            echo -e "${RED}❌ GitHub Pages deployment failed${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}✅ Successfully deployed to GitHub Pages${NC}"
        echo -e "${BLUE}🌍 Documentation available at: https://hyperopt-strat.github.io/platform/${NC}"
        ;;
        
    2)
        echo -e "${YELLOW}👀 Starting local preview server...${NC}"
        echo -e "${BLUE}📱 Documentation will be available at: http://localhost:8000${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
        mkdocs serve --dev-addr=127.0.0.1:8000
        ;;
        
    3)
        echo -e "${YELLOW}🔧 Custom server deployment${NC}"
        read -p "Enter server address (user@host): " server_addr
        read -p "Enter deployment path: " deploy_path
        
        if [ -z "$server_addr" ] || [ -z "$deploy_path" ]; then
            echo -e "${RED}❌ Server address and path are required${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}📤 Uploading to $server_addr:$deploy_path...${NC}"
        
        # Create deployment archive
        tar -czf docs-deploy.tar.gz -C "$BUILD_DIR" .
        
        # Upload and extract
        scp docs-deploy.tar.gz "$server_addr:/tmp/"
        ssh "$server_addr" "mkdir -p '$deploy_path' && cd '$deploy_path' && tar -xzf /tmp/docs-deploy.tar.gz && rm /tmp/docs-deploy.tar.gz"
        
        # Cleanup
        rm docs-deploy.tar.gz
        
        echo -e "${GREEN}✅ Successfully deployed to custom server${NC}"
        ;;
        
    4)
        echo -e "${BLUE}👋 Deployment cancelled${NC}"
        exit 0
        ;;
        
    *)
        echo -e "${RED}❌ Invalid option${NC}"
        exit 1
        ;;
esac

# Post-deployment checks
if [ "$choice" = "1" ]; then
    echo ""
    echo -e "${YELLOW}🔍 Running post-deployment checks...${NC}"
    
    # Wait a moment for deployment to propagate
    sleep 5
    
    # Check if site is accessible
    SITE_URL="https://hyperopt-strat.github.io/platform/"
    if command -v curl &> /dev/null; then
        if curl -s --head "$SITE_URL" | head -n 1 | grep -q "200 OK"; then
            echo -e "${GREEN}✅ Site is accessible at $SITE_URL${NC}"
        else
            echo -e "${YELLOW}⚠️  Site may still be deploying. Check $SITE_URL in a few minutes${NC}"
        fi
    fi
fi

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
if [ -d "$BUILD_DIR" ] && [ "$choice" != "2" ]; then
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}✅ Build directory cleaned${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Documentation deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}📚 Next steps:${NC}"
echo "  - Test the deployed documentation thoroughly"
echo "  - Monitor site performance and accessibility"
echo "  - Set up automated deployment pipelines"
echo "  - Configure domain and SSL certificates"
echo ""
echo -e "${BLUE}📊 Deployment Summary:${NC}"
echo "  - Build size: $BUILD_SIZE"
echo "  - Files generated: $FILE_COUNT"
echo "  - Deployment target: $([ "$choice" = "1" ] && echo "GitHub Pages" || [ "$choice" = "2" ] && echo "Local preview" || echo "Custom server")"
echo ""
echo -e "${GREEN}✨ Happy documenting!${NC}" 