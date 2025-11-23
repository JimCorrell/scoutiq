# Contributing to ScoutIQ

Thank you for your interest in contributing to ScoutIQ! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:

   ```bash
   git clone https://github.com/yourusername/scoutiq.git
   cd scoutiq
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

5. Download required models:
   ```bash
   python -m spacy download en_core_web_lg
   ```

## Making Changes

1. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:

   ```bash
   pytest tests/
   ```

4. Commit your changes:

   ```bash
   git add .
   git commit -m "Description of changes"
   ```

5. Push to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing

- Write unit tests for new functionality
- Ensure all existing tests pass
- Aim for high code coverage

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update SETUP.md for setup-related changes

## Areas for Contribution

- Additional data sources integration
- New NLP features and extractors
- Alternative ML models
- Performance optimizations
- Documentation improvements
- Example notebooks and tutorials
- Bug fixes

## Questions?

Open an issue for discussion before starting major changes.
