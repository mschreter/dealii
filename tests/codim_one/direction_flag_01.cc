//----------------------------  direction_flag_01.cc  ---------------------------
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
//----------------------------  direction_flag_01.cc  ---------------------------

// test direction flags in a 1d mesh embedded in 2d

#include "../tests.h"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

using namespace std;


void test ()
{
  const unsigned int spacedim = 2;
  const unsigned int dim = spacedim-1;

  Triangulation<dim,spacedim> boundary_mesh;
  Triangulation<spacedim> volume_mesh;
  GridGenerator::hyper_cube(volume_mesh);
  GridTools::extract_boundary_mesh (volume_mesh, boundary_mesh);
  for (Triangulation<dim,spacedim>::active_cell_iterator
	 cell = boundary_mesh.begin_active();
       cell != boundary_mesh.end(); ++cell)
    {
      deallog << "Cell=" << cell;
      deallog << ", direction flag="
	      << (cell->direction_flag() ? "true" : "false")
	      << std::endl;
    }

  boundary_mesh.refine_global(1);

  for (Triangulation<dim,spacedim>::active_cell_iterator
	 cell = boundary_mesh.begin_active();
       cell != boundary_mesh.end(); ++cell)
    {
      deallog << "Cell=" << cell << std::endl;
      deallog << ", direction flag="
	      << (cell->direction_flag() ? "true" : "false")
	      << std::endl;
    }
}



int main ()
{
  ofstream logfile("direction_flag_01/output");
  deallog.attach(logfile);
  deallog.depth_console(0);

  test ();
 }
