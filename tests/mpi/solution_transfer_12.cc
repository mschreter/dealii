// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2021 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------



// Verify that SolutionTransfer yields the same result on parallel
// shared Triangulation objects with and without artificial cells.
//
// Test 'pure_refinement'.


#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"


//template <int dim>
//Vector<double>
//refine_and_transfer(const Function<dim>         &function,
                    //Triangulation<dim>          &tria,
                    //const hp::FECollection<dim> &fe_collection,
                    //const unsigned int           active_fe_index)
//{
  //DoFHandler<dim> dh(tria);

  //for (const auto &cell : dh.active_cell_iterators())
  //{
        //if (not cell->is_locally_owned())
          //continue;
    //cell->set_active_fe_index(active_fe_index);
  //}

  //dh.distribute_dofs(fe_collection);

  //tria.begin_active()->set_refine_flag();

//for (const auto &cell : dh.active_cell_iterators())
  //{
        //if (not cell->is_locally_owned())
          //continue;
  //if (cell->refine_flag_set())
    //cell->set_future_fe_index(active_fe_index);
  //}


  //tria.prepare_coarsening_and_refinement();

  //Vector<double> sol_old(dh.n_dofs());
  //VectorTools::interpolate(MappingQ<dim>(1), dh, function, sol_old);

  //SolutionTransfer<dim> soltrans(dh);
  //soltrans.prepare_for_coarsening_and_refinement(sol_old);

  //tria.execute_coarsening_and_refinement();

  //dh.distribute_dofs(fe_collection);

  //Vector<double> sol_new(dh.n_dofs());
  //soltrans.interpolate(sol_new);

  //return sol_new;
//}

using VectorType = LinearAlgebra::distributed::Vector<double>;

template <int dim>
VectorType
refine_and_transfer(const Function<dim>         &function,
                    parallel::shared::Triangulation<dim> &tria,
                    const hp::FECollection<dim> &fe_collection,
                    const unsigned int           active_fe_index)
{
  DoFHandler<dim> dh(tria);

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
    cell->set_active_fe_index(active_fe_index);

  dh.distribute_dofs(fe_collection);

  IndexSet locally_owned_dofs = dh.locally_owned_dofs();

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dh, locally_relevant_dofs);

  VectorType sol_old;
  sol_old.reinit(locally_owned_dofs,
                 locally_relevant_dofs,
                 MPI_COMM_WORLD);

  VectorTools::interpolate(MappingQ<dim>(1), dh, function, sol_old);
  sol_old.update_ghost_values();

  tria.begin_active()->set_refine_flag();

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      if (cell->refine_flag_set())
        cell->set_future_fe_index(active_fe_index);

  tria.prepare_coarsening_and_refinement();

  std::vector<const VectorType *> old_vectors;
  old_vectors.push_back(&sol_old);

  SolutionTransfer<dim, VectorType> soltrans(dh);
  soltrans.prepare_for_coarsening_and_refinement(old_vectors);

  tria.execute_coarsening_and_refinement();

  dh.distribute_dofs(fe_collection);

  locally_owned_dofs = dh.locally_owned_dofs();
  locally_relevant_dofs.clear();
  DoFTools::extract_locally_relevant_dofs(dh, locally_relevant_dofs);

  VectorType sol_new;
  sol_new.reinit(locally_owned_dofs,
                 locally_relevant_dofs,
                 MPI_COMM_WORLD);
  sol_new = 0.0;
  sol_new.zero_out_ghost_values();

  std::vector<VectorType *> new_vectors;
  new_vectors.push_back(&sol_new);

  sol_new.zero_out_ghost_values();
  soltrans.interpolate(new_vectors);
  sol_new.update_ghost_values();

  return sol_new;
}


template <int dim>
void
test(const Function<dim>         &function,
     const hp::FECollection<dim> &fe_collection,
     const unsigned int           active_fe_index)
{
  parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD,
                                            Triangulation<dim>::none,
                                            false);
  parallel::shared::Triangulation<dim> tria_artificial(MPI_COMM_WORLD,
                                                       Triangulation<dim>::none,
                                                       true);

  GridGenerator::subdivided_hyper_cube(tria, 2);
  GridGenerator::subdivided_hyper_cube(tria_artificial, 2);

  VectorType sol =
    refine_and_transfer(function, tria, fe_collection, active_fe_index);

  VectorType sol_artificial =
    refine_and_transfer(function, tria_artificial, fe_collection, active_fe_index);

  AssertDimension(sol.size(), sol_artificial.size());
VectorType difference(sol);
difference -= sol_artificial;

AssertThrow(difference.linfty_norm() < 1e-14, ExcInternalError());
  deallog << "OK" << std::endl;
}


template <int dim>
void
run_tests_for_dimension()
{
  {
    deallog.push("FENothing_FEQ");

    hp::FECollection<dim> fe_collection;
    fe_collection.push_back(FE_Nothing<dim>());
    fe_collection.push_back(FE_Q<dim>(1));

    test<dim>(Functions::CosineFunction<dim>(), fe_collection, 1);

    deallog.pop();
  }

  {
    deallog.push("FEQ_FEQ");

    hp::FECollection<dim> fe_collection;
    fe_collection.push_back(FE_Q<dim>(1));
    fe_collection.push_back(FE_Q<dim>(1));

    test<dim>(Functions::CosineFunction<dim>(), fe_collection, 0);

    deallog.pop();
  }

  {
    deallog.push("FEQ_FENothing");

    hp::FECollection<dim> fe_collection;
    fe_collection.push_back(FE_Q<dim>(1));
    fe_collection.push_back(FE_Nothing<dim>());

    test<dim>(Functions::CosineFunction<dim>(), fe_collection, 0);

    deallog.pop();
  }
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    log;

  //deallog.push("1d");
  //run_tests_for_dimension<1>();
  //deallog.pop();

  deallog.push("2d");
  run_tests_for_dimension<2>();
  deallog.pop();

  deallog.push("3d");
  run_tests_for_dimension<3>();
  deallog.pop();
}
