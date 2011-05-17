//----------------------------  rt_5.cc  ---------------------------
//    rt_5.cc,v 1.1 2003/06/09 15:59:07 wolf Exp
//    Version: 
//
//    Copyright (C) 2003, 2004, 2005, 2006, 2008, 2009, 2010 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------  rt_5.cc  ---------------------------


// Just output the restriction matrices of the RT element

#include "../tests.h"
#include <deal.II/base/logstream.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <fstream>
#include <string>

#define PRECISION 5



template<int dim>
void
test(const unsigned int degree)
{
  deallog << "FE_RaviartThomas<" << dim << "> (" << degree << ")"
	  << std::endl;
  
  FE_RaviartThomas<dim> fe_rt(degree);

  for (unsigned int c=0; c<GeometryInfo<dim>::max_children_per_cell; ++c)
    {
      const FullMatrix<double> & m = fe_rt.get_restriction_matrix(c);

      for (unsigned int i=0; i<m.m(); ++i)
	{
	  for (unsigned int j=0; j<m.n(); ++j)
	    deallog << m(i,j) << ' ';
	  deallog << std::endl;
	}
      
      deallog << std::endl;
    }
}


int
main()
{
  std::ofstream logfile ("rt_5/output");
  deallog << std::setprecision(PRECISION);
  deallog << std::fixed;  
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);

  for (unsigned int degree=0; degree<3; ++degree)
    {
      test<2>(degree);
      test<3>(degree);
    }
  
  return 0;
}
