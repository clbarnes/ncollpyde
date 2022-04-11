#!/bin/sh
# Symlink all cargo-installed binaries into
py_bin_dir="$(python -c "import sys, os; print(os.path.dirname(sys.executable))")"
cargo_bin_dir="$HOME/.cargo/bin"
for name in $(ls $cargo_bin_dir); do
    orig="$cargo_bin_dir/$name"
    lnk="$py_bin_dir/$name"
    echo "Symlinking $orig to $lnk"
    ln -s $orig $lnk
done

