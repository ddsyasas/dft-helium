# How to Use the Helium DFT Program — A Simple Guide

This guide is for a person — a student, a teacher, anyone curious — who wants to
**run this program, get the results and graphs, and understand what they mean**.
No prior coding needed beyond running cells in a notebook. Read top to bottom the
first time; after that, jump to the part you need.

---

## 1. What this program actually does

It calculates **how tightly the two electrons in a helium atom are bound to the
nucleus** — a single number called the *ground-state energy*. The real answer
from experiment is about **−2.90 Hartree** (Hartree is just the energy unit we
use here; more negative = more tightly bound).

Doing this exactly by hand is impossible, because the two electrons push on each
other. So the program builds the answer up in steps, adding one piece of physics
at a time, and you watch the number get closer to the truth.

**The one idea to hold onto:** instead of tracking two electrons separately, we
track the **electron cloud** — how much electron "stuff" sits at each distance
from the nucleus. Everything is computed from that cloud.

---

## 2. A little theory, in plain words

- **The nucleus pulls the electrons in.** This pull gets stronger closer to the
  centre. We write it as a "potential well."
- **The electrons push on each other.** This push is called the **Hartree**
  term. It works against the nuclear pull.
- **Two quantum corrections** make the electrons avoid each other a bit more than
  simple pushing would suggest. These are **exchange** and **correlation**.
- **The circular problem:** where the electrons settle depends on the pushes and
  pulls, but the pushes depend on where the electrons are. The program solves
  this by **guessing, solving, updating, and repeating until the answer stops
  changing.** That settled answer is what we report. This loop is called
  *self-consistency*.

That's the whole story. Each time we add a piece (push → exchange → correlation),
the energy improves.

---

## 3. What you need (one-time setup)

You need Python with two libraries: `numpy` and `matplotlib`. To run the
notebook and the checks you also want `jupyter` and `pytest`. If they are not
installed, run this once in a terminal:

```bash
pip install numpy matplotlib jupyter pytest
```

The project has three things you will touch:

- **`helium_dft.ipynb`** — the notebook. This is where you *do everything* and
  *see everything*. Start here.
- **`dft.py`** — the engine (all the calculations live here). You usually don't
  edit it; the notebook calls it for you.
- **`figures/`** — where the saved picture files end up.

---

## 4. How to run it (the main way)

1. Open a terminal in the project folder.
2. Start Jupyter:
   ```bash
   jupyter notebook helium_dft.ipynb
   ```
   (or open the file in VS Code / JupyterLab — anything that runs notebooks).
3. Run the cells **from the top, one by one, in order** (press `Shift+Enter` on
   each cell), or use the menu: *Run → Run All Cells*.

**Why in order?** Each cell builds on the one before it. The grid is made first,
then the simple checks, then the full helium calculation, then the graphs. If you
jump ahead, a later cell won't find what it needs.

> **Note on the "kernel":** the notebook is set to run with a kernel named
> *"Python 3.13 (dft)"*. If the notebook complains it can't find a kernel, pick
> your normal Python 3 kernel from the *Kernel → Change Kernel* menu — as long as
> that Python has `numpy` and `matplotlib`, it will work.

---

## 5. What you enter (the inputs)

There is a small **parameters block** near the top of the notebook. These are the
only knobs, and the defaults already give the right answers:

| You enter | Means | Default | Try changing it to…|
|---|---|---|---|
| `Z` | the nuclear charge | `2` (helium) | `1` to get hydrogen instead |
| `H_STEP` | how fine the grid is (smaller = more accurate, slower) | `0.001` | `0.005` to see it run faster but slightly less accurate |
| `R_MAX` | how far out we calculate (in Bohr) | `25` | usually leave alone |

For helium you do not need to change anything — just run the cells.

---

## 6. What you see, step by step (and what it means)

As you run down the notebook, here is what appears and how to read it.

### Step A — Setup check
**You see:** a line like `Grid: N = 25000 points ...` and a "Smoke test" line
where an integral comes out as ≈ 1.0.
**Means:** the grid and the basic math tools are working. If this prints with no
red error, you're good to continue.

### Step B — One electron, the easy case
**You see:** an energy of about **−0.5** for hydrogen and **−2.0** for a helium
nucleus with a single electron, each next to its target, marked **PASS**. Then a
graph of the wavefunction `u(r)` sitting exactly on top of the known exact curve.
**Means:** the core solver is correct. We can trust it on the hard case.
**What `u(r)` is:** the shape of the electron cloud as you move out from the
nucleus. It starts at zero at the centre, rises to a peak, then fades to zero far
away.

### Step C — The electron-cloud push (Hartree)
**You see:** two curves matching exact formulas, marked **PASS**.
**Means:** the program correctly computes the repulsion the cloud creates.

### Step D — Helium with the basic push only
**You see:** a table of numbers printed each loop, the energy settling down to
about **−2.86**, marked **PASS**, plus a convergence graph (energy vs loop
number) flattening out, and a picture of the final electron cloud in its
potential well.
**Means:** the guess-solve-update loop has converged. The flat line on the graph
is what "self-consistent" looks like — the answer stopped changing.

### Step E — Add the exchange correction
**You see:** the loop again, settling to about **−2.72**.
**Means:** exchange has been added. (It's normal that this number is a little
*less* bound than Step D — see the note in Section 8.)

### Step F — Add the correlation correction (the full result)
**You see:** the loop settling to about **−2.83**, then a final "energy ladder"
summary printing all the numbers together.
**Means:** this is the headline result — the best this method gives, **−2.83**,
close to the exact **−2.90**.

### Step G — The report figures and table
**You see:** four polished graphs and a printed results table, plus messages like
`saved: figures/fig1_scf_convergence.png ...`.
**Means:** the picture files and the table have been written to the `figures/`
folder, ready to drop into a report.

---

## 7. The graphs and how to read them

After running, four saved pictures appear in the `figures/` folder (each as a
`.png` for viewing and a `.pdf` for printing):

- **`fig1_scf_convergence`** — energy vs loop number for all three models on one
  chart. Read it left to right: the lines start off, wobble, then go flat. *Flat
  = converged = trustworthy.* The dashed line is the exact answer.
- **`fig2_orbital_density`** — the final electron cloud. Left: the cloud shape.
  Right: how dense the electrons are at each distance (note most of the cloud sits
  within ~2 Bohr of the nucleus).
- **`fig3_potentials`** — the "energy landscape" the electrons feel, broken into
  its pieces: the deep nuclear pull, the repulsive cloud push, and the small
  exchange and correlation corrections.
- **`fig4_energy_comparison`** — a bar chart comparing your computed numbers to
  the standard reference values, with the exact value marked.

To view them, just open the `figures/` folder and double-click a `.png`.

---

## 8. The numbers you should get (and what they say)

| What is included | Energy (Ha) | Reading |
|---|---:|---|
| Basic push only | **−2.86** | already a good answer |
| + exchange | **−2.72** | *less* bound — see note below |
| + correlation (full result) | **−2.83** | the best this method gives |
| Exact (for comparison) | **−2.90** | the target |

**Why does adding exchange make it look "worse" (−2.72)?** The simplest model
(−2.86) happens to match a famous benchmark (Hartree–Fock) because, for helium's
two electrons, that benchmark and the simple model agree exactly. The exchange
recipe we add next is an *approximation*, so on its own it overshoots a little.
Adding correlation then pulls the answer back to −2.83. This back-and-forth is
expected and is a good thing to explain in a write-up.

---

## 9. Want a quick number without the whole notebook?

You can get the three energies in a few lines. Open a Python prompt in the
project folder and type:

```python
import dft
grid = dft.make_grid()                          # build the radial grid
print(dft.scf_no_xc(Z=2.0, r=grid, verbose=False)["E"])   # ≈ -2.86
print(dft.scf_lda_x (Z=2.0, r=grid, verbose=False)["E"])  # ≈ -2.72
print(dft.scf_lda_xc(Z=2.0, r=grid, verbose=False)["E"])  # ≈ -2.83
```

Set `verbose=True` if you want to watch each loop print as it converges.

To check a single electron instead:

```python
energy, cloud = dft.find_eigenvalue(Z=1.0, r=grid)   # hydrogen
print(energy)                                        # ≈ -0.5
```

---

## 10. Checking everything still works

There is a built-in set of checks. From a terminal in the project folder:

```bash
pytest -v
```

**You see:** eight lines, each ending in `PASSED`.
**Means:** every result the program is supposed to reproduce still matches its
known target. If you ever change something and a line says `FAILED`, that result
no longer matches — undo your change. (The full helium runs take a couple of
minutes, so be patient.)

---

## 11. If something goes wrong

- **A cell shows a red error about a missing library** → run the `pip install`
  line from Section 3, then restart the notebook (*Kernel → Restart*).
- **"No bound state found"** → you changed an input too far. Put `Z`, `H_STEP`,
  and `R_MAX` back to their defaults (`2`, `0.001`, `25`).
- **A later cell errors saying a name is undefined** → you skipped a cell. Run
  from the top again (*Run → Run All Cells*).
- **The energy doesn't settle / the loop number gets very large** → reduce the
  step size of the update (the `w` value, default `0.3`) toward `0.2`.

---

## 12. Things to try next (small experiments)

- **Switch to hydrogen:** set `Z = 1` and rerun the single-electron check — you
  should get exactly −0.5.
- **See accuracy vs speed:** set `H_STEP = 0.005`, rerun, and watch the helium
  numbers shift slightly — finer grids cost time but buy accuracy.
- **Watch convergence live:** run the helium cells with `verbose=True` and follow
  the energy column shrinking toward its final value each loop.
- **Read off the cloud:** from `fig2`, estimate how far from the nucleus the
  electrons mostly live (hint: look where the right-hand curve peaks).

---

### One-line recap
Open the notebook → run the cells top to bottom → read the printed energies
(−2.86, then −2.72, then −2.83) and the four graphs in `figures/` → compare to
the exact −2.90. That's the whole workflow.
