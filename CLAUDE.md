# Knowledge Base Project Instructions

This is a personal knowledge management repository using Obsidian + Git for long-term learning and research.

## Repository Structure

```
knowledge-base/
├── atomic/          # Single-concept notes (core knowledge)
├── projects/        # Project-specific work (thesis, courses, etc.)
├── templates/       # Note templates for consistency
└── claude.md        # This file
```

## Key Principles

### 1. Atomic Notes

- One concept per note
- Stored in `atomic/`
- Use template from `templates/atomic_note.md`
- Link concepts using `[[note_name]]` syntax

### 2. Note Naming Convention

- Use lowercase with underscores: `reml_criterion.md`
- Be descriptive but concise
- No dates in filenames (use frontmatter)

### 3. Linking Strategy

- **Links (`[[]]`)**: For specific concept relationships
- **Tags (`#`)**: For broad categorization
- Always connect new concepts to existing notes

## When Working With Me

### Creating New Notes

When I create atomic notes:

1. Use the template structure from `templates/atomic_note.md`
2. Include proper frontmatter (date, tags)
3. Suggest connections to existing notes
4. Keep explanations clear but concise

### Code Files

When creating code implementations:

1. Place in `projects/[project-name]/code/`
2. Include docstrings explaining the concept
3. Link to relevant atomic notes in comments

### Project Work

For any project:

1. Create project folder: `projects/[project-name]/`
2. Reference atomic notes for reusable concepts
3. Keep project-specific notes separate from general knowledge

## Git Workflow

### Commit Message Format

```
Type: Brief description

Types:
- Add: New note/concept
- Update: Revisions to existing notes
- Project: Project-specific work
- Code: Implementation work
```

### Examples

- `Add: REML criterion derivation`
- `Update: Clarify penalized regression explanation`
- `Project: Thesis chapter 2 draft`

## Note Status Convention

Use these status indicators in notes:

- 🌱 **seedling**: Initial capture, needs development
- 🌿 **growing**: Partial understanding, actively developing
- 🌳 **evergreen**: Well-understood, stable reference

## Commands You Can Run

When I create files, I'll place them in appropriate locations:

- Atomic concepts → `atomic/[topic_name].md`
- Code implementations → `projects/[project]/code/`
- Templates → `templates/`

## Current Active Projects

List your active projects here as you work on them.

## Instructions for Creating Content

### When I Ask for Explanations

1. Create atomic note in appropriate location
2. Start with simple explanation
3. Build up mathematical detail incrementally
4. Include code examples when relevant
5. Suggest related concepts to explore next

### When Writing Code

1. Explain what the code does conceptually first
2. Reference the mathematical formulation from atomic notes
3. Include comments linking to note files
4. Keep functions small and well-documented

### Pacing

- Go step-by-step, waiting for confirmation before proceeding
- Present one logical section at a time
- Explain the "why" before the "how"
- Build up complexity gradually

## Do Not

- Don't create nested folder structures without asking
- Don't dump large amounts of information at once
- Don't skip explanations of mathematical concepts
- Don't create files outside the defined structure without discussion

## File Creation Workflow

When creating a new atomic note:

1. Check if related concepts already exist
2. Use template structure
3. Add frontmatter with date and initial tags
4. Write "What is this?" summary first
5. Add detailed notes
6. List connections to other notes
7. Include questions for future exploration

## Repository Purpose

This is a long-term knowledge repository for:

- Learning and research notes
- Concept development across multiple domains
- Project work that builds on reusable knowledge
- Code implementations with clear explanations

The repository uses:

- **Obsidian** for note-taking and linking concepts
- **Git** for version control and long-term storage
- **Atomic note structure** for building connected knowledge over time