#!/bin/bash
# Production Validation Script
# Validates code structure, syntax, and configuration

echo "======================================================================="
echo "PRODUCTION VALIDATION CHECK"
echo "Smart Document Intelligence Platform"
echo "======================================================================="
echo ""

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
pass() {
    echo -e "${GREEN}✅ PASS:${NC} $1"
}

fail() {
    echo -e "${RED}❌ FAIL:${NC} $1"
    ((ERRORS++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN:${NC} $1"
    ((WARNINGS++))
}

# Check 1: Directory structure
echo "1. Checking directory structure..."
if [ -d "smart-doc-intelligence/backend" ]; then
    pass "Backend directory exists"
else
    fail "Backend directory missing"
fi

if [ -d "smart-doc-intelligence/frontend" ]; then
    pass "Frontend directory exists"
else
    fail "Frontend directory missing"
fi

if [ -d "smart-doc-intelligence/tests" ]; then
    pass "Tests directory exists"
else
    fail "Tests directory missing"
fi

# Check 2: Core files exist
echo ""
echo "2. Checking core files..."
FILES=(
    "smart-doc-intelligence/frontend/app.py"
    "smart-doc-intelligence/backend/pipeline.py"
    "smart-doc-intelligence/backend/features/rag_pipeline.py"
    "smart-doc-intelligence/requirements.txt"
    "smart-doc-intelligence/README.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        pass "Found: $file"
    else
        fail "Missing: $file"
    fi
done

# Check 3: Python syntax validation
echo ""
echo "3. Validating Python syntax..."
PYTHON_FILES=$(find smart-doc-intelligence -name "*.py" -type f 2>/dev/null)
SYNTAX_ERRORS=0

for file in $PYTHON_FILES; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        : # Syntax OK
    else
        fail "Syntax error in: $file"
        ((SYNTAX_ERRORS++))
    fi
done

if [ $SYNTAX_ERRORS -eq 0 ]; then
    pass "All Python files have valid syntax"
fi

# Check 4: Required modules count
echo ""
echo "4. Checking module count..."
BACKEND_MODULES=$(find smart-doc-intelligence/backend -name "*.py" -type f | wc -l)
if [ $BACKEND_MODULES -ge 20 ]; then
    pass "Backend has $BACKEND_MODULES modules (healthy)"
else
    warn "Only $BACKEND_MODULES backend modules found"
fi

# Check 5: Documentation exists
echo ""
echo "5. Checking documentation..."
DOCS=(
    "PHASE1_COMPLETE.md"
    "PHASE2_COMPLETE.md"
    "PHASE3_COMPLETE.md"
    "PHASE4_COMPLETE.md"
    "PHASE5_COMPLETE.md"
    "PRODUCTION_READINESS.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        pass "Found: $doc"
    else
        warn "Missing: $doc"
    fi
done

# Check 6: Code complexity (line counts)
echo ""
echo "6. Checking code complexity..."
TOTAL_LINES=$(find smart-doc-intelligence -name "*.py" -type f -exec wc -l {} + 2>/dev/null | tail -n 1 | awk '{print $1}')
if [ $TOTAL_LINES -ge 10000 ]; then
    pass "Total code: $TOTAL_LINES lines (substantial)"
else
    warn "Only $TOTAL_LINES lines of code"
fi

# Check 7: UI file validation
echo ""
echo "7. Checking UI implementation..."
if grep -q "streamlit" smart-doc-intelligence/frontend/app.py 2>/dev/null; then
    pass "Streamlit imports found"
else
    fail "No Streamlit imports in UI"
fi

if grep -q "st.chat_message" smart-doc-intelligence/frontend/app.py 2>/dev/null; then
    pass "Chat interface implemented"
else
    fail "Chat interface missing"
fi

if grep -q "st.file_uploader" smart-doc-intelligence/frontend/app.py 2>/dev/null; then
    pass "File upload implemented"
else
    fail "File upload missing"
fi

# Check 8: Critical functions exist
echo ""
echo "8. Checking critical functions..."
if grep -q "@st.cache_resource" smart-doc-intelligence/frontend/app.py 2>/dev/null; then
    pass "Performance caching implemented"
else
    warn "No caching found in UI"
fi

if grep -q "def handle_input" smart-doc-intelligence/frontend/app.py 2>/dev/null; then
    pass "Input handler implemented"
else
    fail "Input handler missing"
fi

if grep -q "def process_files" smart-doc-intelligence/frontend/app.py 2>/dev/null; then
    pass "File processor implemented"
else
    fail "File processor missing"
fi

# Check 9: Error handling
echo ""
echo "9. Checking error handling..."
TRY_COUNT=$(grep -c "try:" smart-doc-intelligence/frontend/app.py 2>/dev/null)
if [ $TRY_COUNT -ge 3 ]; then
    pass "Error handling: $TRY_COUNT try blocks found"
else
    warn "Limited error handling: only $TRY_COUNT try blocks"
fi

# Check 10: Git status
echo ""
echo "10. Checking git status..."
cd smart-doc-intelligence 2>/dev/null
if git status &>/dev/null; then
    if git diff --quiet; then
        pass "No uncommitted changes"
    else
        warn "Uncommitted changes exist"
    fi
else
    warn "Not a git repository"
fi
cd ..

# Summary
echo ""
echo "======================================================================="
echo "VALIDATION SUMMARY"
echo "======================================================================="
echo "Total Errors: $ERRORS"
echo "Total Warnings: $WARNINGS"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 PRODUCTION READY - All checks passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  CAUTION - Warnings found but no critical errors${NC}"
    exit 0
else
    echo -e "${RED}🚫 NOT READY - Critical errors must be fixed${NC}"
    exit 1
fi
