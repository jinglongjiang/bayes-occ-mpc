set -e
P=/home/abc/workspace/bayes_occ_mpc/build_modern/gsl
mkdir -p $P/src && cd $P/src
if [ ! -f gsl-2.7.1.tar.gz ]; then
  curl -fL --retry 3 -o gsl-2.7.1.tar.gz https://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz
fi
sha256sum gsl-2.7.1.tar.gz
[ -d gsl-2.7.1 ] || tar xf gsl-2.7.1.tar.gz
cd gsl-2.7.1
./configure --prefix=$P --enable-shared --disable-static > $P/configure.log 2>&1
make -j4 > $P/make.log 2>&1
make install > $P/install.log 2>&1
echo "=== GSL_OK ==="
ls $P/lib/pkgconfig/gsl.pc && pkg-config --modversion --define-prefix $P/lib/pkgconfig/gsl.pc 2>/dev/null || cat $P/lib/pkgconfig/gsl.pc | head -5
