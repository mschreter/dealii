//-----------------------------------------------------------------------------
//    $Id$
//    Version: $Name$
//
//    Copyright (C) 2010 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//-----------------------------------------------------------------------------

// test IndexSet::read() and write()

#include "../tests.h"
#include <iomanip>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <stdlib.h>

#include <deal.II/base/index_set.h>


void test ()
{
  IndexSet is1 (100);

  is1.add_range(0,10);
  is1.add_range(20,100);

  std::ofstream out("index_set_readwrite_01/a.idxset");
  is1.write(out);
  IndexSet is2;
  std::ifstream in("index_set_readwrite_01/a.idxset");
  is2.read(in);

  Assert(is1 == is2, ExcInternalError());


  IndexSet is3(11);
  is3.add_range(3,5);
  std::ifstream in2("index_set_readwrite_01/a.idxset");
  is3.read(in2);

  Assert(is1 == is3, ExcInternalError());

  deallog << "OK" << std::endl;

  std::remove ("index_set_readwrite_01/a.idxset");
}




int main()
{
  std::ofstream logfile("index_set_readwrite_01/output");
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);

  test ();
}
