//----------------------------  compressed_set_sparsity_pattern_06.cc  ---------------------------
//    $Id$
//    Version: $Name$ 
//
//    Copyright (C) 2004, 2005 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------  compressed_set_sparsity_pattern_06.cc  ---------------------------


// check CompressedSetSparsityPattern::print_gnuplot. since we create quite some
// output here, choose smaller number of rows and entries than in the other
// tests

#include "../tests.h"
#include <deal.II/base/logstream.h>
#include <deal.II/lac/compressed_set_sparsity_pattern.h>
#include <iomanip>
#include <fstream>


std::ofstream logfile("compressed_set_sparsity_pattern_06/output");

void test ()
{
  const unsigned int N = 100;
  CompressedSetSparsityPattern csp (N,N);
  for (unsigned int i=0; i<N; ++i)
    for (unsigned int j=0; j<10; ++j)
      csp.add (i, (i+(i+1)*(j*j+i))%N);
  csp.symmetrize ();

  csp.print_gnuplot (logfile);
  deallog << "OK" << std::endl;
}



int main () 
{
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);

  test ();
  return 0;
}
