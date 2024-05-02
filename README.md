```bash
git clone -o template https://github.com/gzbfgjf2/template.git my-project && cd my-project
git switch -c main
python -m venv .venv
. .venv/bin/activate
pip install -e .
# develop your repo
git remote add origin <my-repo-remote-url>
git push -u origin main
```
