# 🤝 WORKING WITH CLAUDE CODE - QUICK REFERENCE

## 💡 THE WORKFLOW

**You (Browser Claude):**
- 📋 Create specifications and plans
- 🎯 Define requirements and architecture  
- 📝 Write detailed documentation
- 🧠 Strategic thinking and design

**Claude Code (Terminal):**
- ⚙️ Implement the actual code
- 🔧 Modify files and functions
- 🧪 Run tests and verify
- 🚀 Execute the build tasks

---

## 📦 FILES CREATED FOR CLAUDE CODE

### 1. **CLAUDE_CODE_IMPLEMENTATION_SPEC.md** 
Complete technical specification with:
- Exact code snippets to add
- File locations and modifications
- Method signatures and logic
- Testing procedures

### 2. **CLAUDE_CODE_COMMANDS.md**
Ready-to-use commands for Claude Code:
- Copy/paste prompts
- Step-by-step tasks
- Test commands
- Success criteria

### 3. **MARKET_DATA_UI_INTEGRATION_PLAN.md**
High-level overview:
- Architecture decisions
- Implementation phases
- Priority order
- Technical notes

---

## 🎯 HOW TO USE

### Simple Approach (Recommended):
```bash
# In your terminal with Claude Code:
claude "Read CLAUDE_CODE_COMMANDS.md and execute Command 1"
```

### Detailed Approach:
```bash
# Point Claude Code to the spec:
claude "Read CLAUDE_CODE_IMPLEMENTATION_SPEC.md and implement Task 1: Level 2 Market Depth. Update ibkr_connector.py, dashboard_api.py, and customizable_platform.html as specified."
```

### All-at-Once Approach:
```bash
# Do everything:
claude "Read CLAUDE_CODE_IMPLEMENTATION_SPEC.md and implement all 4 tasks in priority order. Test each component after implementation."
```

---

## 🔄 ITERATION WORKFLOW

1. **You define what needs to be built** (in this chat)
2. **I create detailed specs** (saved to /outputs/)
3. **You give specs to Claude Code** (in terminal)
4. **Claude Code implements** (modifies your files)
5. **You test and report back** (in this chat)
6. **We iterate and refine** (repeat as needed)

---

## 📁 FILE LOCATIONS

All specs are in: `C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2\`

Download from this chat:
- [CLAUDE_CODE_IMPLEMENTATION_SPEC.md](computer:///mnt/user-data/outputs/CLAUDE_CODE_IMPLEMENTATION_SPEC.md)
- [CLAUDE_CODE_COMMANDS.md](computer:///mnt/user-data/outputs/CLAUDE_CODE_COMMANDS.md)
- [MARKET_DATA_UI_INTEGRATION_PLAN.md](computer:///mnt/user-data/outputs/MARKET_DATA_UI_INTEGRATION_PLAN.md)

---

## 💬 SAMPLE CONVERSATIONS

### With Me (Browser):
**You:** "I need to add order execution to the platform"  
**Me:** *Creates detailed spec with endpoints, error handling, order types*  
**You:** "Great! I'll have Claude Code build it"

### With Claude Code (Terminal):
**You:** `claude "Read the order execution spec and implement it"`  
**Claude Code:** *Modifies files, adds functions, tests code*  
**Claude Code:** "Done! Created 3 endpoints and updated UI. Run tests."

### Back to Me (Browser):
**You:** "Claude Code finished order execution. Need help testing it."  
**Me:** *Creates test suite and validation checklist*

---

## 🎨 DIVISION OF LABOR

### I'm Great At:
- ✅ Architecture and design
- ✅ Writing specifications
- ✅ Creating documentation
- ✅ Strategic planning
- ✅ Explaining concepts
- ✅ Troubleshooting issues

### Claude Code is Great At:
- ✅ Implementing code quickly
- ✅ Modifying multiple files
- ✅ Following specifications exactly
- ✅ Running tests
- ✅ Executing terminal commands
- ✅ Rapid iteration

---

## 🚀 GETTING STARTED NOW

### Step 1: Get the Specs
Download the 3 files linked above and save them to your project folder.

### Step 2: Choose Your Task
Look at CLAUDE_CODE_COMMANDS.md and pick:
- Command 1: Level 2 (recommended start)
- Command 2: Charts
- Command 3: Account Data
- Command 4: Time & Sales
- Command 5: All at once

### Step 3: Tell Claude Code
```bash
claude "Read CLAUDE_CODE_COMMANDS.md and execute Command 1"
```

### Step 4: Report Back
Come back here and tell me:
- What Claude Code implemented
- Any errors or issues
- What to build next

---

## 🎯 EXAMPLE SESSION

**You (to me in browser):**
> "Let's integrate Level 2 market depth"

**Me:**
> *Creates detailed spec with code snippets*

**You (to Claude Code in terminal):**
> `claude "Implement Level 2 market depth per CLAUDE_CODE_IMPLEMENTATION_SPEC.md Task 1"`

**Claude Code:**
> *Updates ibkr_connector.py, dashboard_api.py, customizable_platform.html*
> "Implementation complete. Test with: curl http://127.0.0.1:9101/api/level2/AAPL"

**You (testing):**
> `curl http://127.0.0.1:9101/api/level2/AAPL`
> *See real bid/ask data!*

**You (back to me):**
> "Level 2 works! Now let's do charts."

**Me:**
> "Great! Tell Claude Code to run Command 2..."

---

## 🎉 BENEFITS OF THIS WORKFLOW

✅ **Clear separation of concerns**
- I handle strategy
- Claude Code handles implementation

✅ **Detailed specifications**
- No ambiguity
- Exact code provided
- Easy to follow

✅ **Fast iteration**
- Claude Code implements quickly
- You test immediately
- I help troubleshoot

✅ **Quality output**
- Well-planned architecture
- Consistent implementation
- Tested components

---

## 📞 WHEN TO TALK TO WHO

### Talk to Me When:
- Planning new features
- Designing architecture
- Need explanations
- Troubleshooting errors
- Creating documentation
- Strategic decisions

### Talk to Claude Code When:
- Ready to implement
- Need code written
- Modifying files
- Running tests
- Executing builds
- Quick iterations

---

## ✅ CURRENT STATUS

**Done:**
- ✅ Detailed implementation specs created
- ✅ Claude Code commands prepared
- ✅ Integration plan documented

**Ready for Claude Code:**
- 🎯 Level 2 Market Depth
- 🎯 Historical Chart Data
- 🎯 Account Data Integration
- 🎯 Time & Sales Tape

**Next Steps:**
1. Download the 3 spec files
2. Choose a starting task
3. Give the spec to Claude Code
4. Test the implementation
5. Report back for next iteration!

---

**🚀 You're all set! Give Claude Code one of those commands and watch it build!**
