//----------------------------  logtest.cc  ---------------------------
//    $Id$
//    Version: $Name$ 
//
//    Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------  logtest.cc  ---------------------------


// some tests for the logstream class, writing output, pushing and
// popping levels, etc


#include "../tests.h"
#include <deal.II/base/logstream.h>
#include <fstream>
#include <iomanip>
#include <limits>


int main()
{
  std::ofstream logfile("logtest/output");
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);

  deallog << "Test" << std::endl;
  deallog.push("l1");
  deallog << "Test1" << std::endl;
  deallog.push("l2");
  deallog << "Test2" << "Test3" << std::endl;
  deallog.push("l3");
  deallog << "Test4";
  deallog.pop();
  deallog << "Test5" << std::endl;
  deallog.pop();
  deallog << "Test6" << std::endl;
  deallog.pop();
  deallog << "Test7" << std::endl;
}
