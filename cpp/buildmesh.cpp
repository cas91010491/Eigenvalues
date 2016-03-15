#include <stdio.h>
#include <vector>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>

namespace dolfin {

Mesh build_mesh(const std::vector<unsigned int>& cells, const std::vector<double>& vertices, int dim)
{
    // vertices and cells are flattened
    unsigned int vlen = vertices.size() / dim;
    unsigned int clen = cells.size() / (dim + 1);

    Mesh mesh;
    
    MeshEditor editor;
    editor.open(mesh, dim, dim);
    editor.init_vertices(vlen);
    editor.init_cells(clen);
    if (dim==3)
    {
        for (int i=0; i<vlen; i++)
	    editor.add_vertex(i, vertices[3*i], vertices[3*i+1], vertices[3*i+2]);
    	for (int i=0; i<clen; i++)
	    editor.add_cell(i, cells[4*i], cells[4*i+1], cells[4*i+2], cells[4*i+3]);
    }
    else
    {
        for (int i=0; i<vlen; i++)
	    editor.add_vertex(i, vertices[2*i], vertices[2*i+1]);
    	for (int i=0; i<clen; i++)
	    editor.add_cell(i, cells[3*i], cells[3*i+1], cells[3*i+2]);
    }
    editor.close();

    return mesh;
}

};
