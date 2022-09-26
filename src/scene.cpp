#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "TinyObjLoader/tiny_obj_loader.h"

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom();
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

Scene::~Scene() {

}

int Scene::loadGeom() {
    int objectid = geoms.size();
    cout << "Loading Geom " << objectid << "..." << endl;
    Geom newGeom;
    string line;

    //load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        if (strcmp(line.c_str(), "sphere") == 0) {
            cout << "Creating new sphere..." << endl;
            newGeom.type = SPHERE;
        } else if (strcmp(line.c_str(), "cube") == 0) {
            cout << "Creating new cube..." << endl;
            newGeom.type = CUBE;
        } else {
            newGeom.type = MESH;

            // mesh objects are in the fomat: [file type] [path to file]
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (tokens.size() != 2) {
                cerr << "ERROR: unrecognized object type\nat line: " << line << endl;
                return -1;
            }
            if (tokens[0] == "obj") {
                cout << "Loading obj mesh " << tokens[1] << endl;
                size_t pos = tokens[1].find_last_of('/');
                if (pos == string::npos) {
                    cerr << "ERROR: invalid obj file path: " << tokens[1] << endl;
                    return -1;
                }
                
                // default material folder to the folder where the mesh is in
                tinyobj::ObjReaderConfig config;
                config.mtl_search_path = tokens[1].substr(0, pos);
                cout << "set material lookup path to: " << config.mtl_search_path << endl;
                

                tinyobj::ObjReader reader;
                if (!reader.ParseFromFile(tokens[1], config)) {
                    if (!reader.Error().empty()) {
                        cerr << "TinyObjReader: ERROR: \n";
                        cerr << reader.Error() << endl;
                    } else {
                        cerr << "no idea what the hell is happening\n";
                    }
                    return -1;
                }
                if (!reader.Warning().empty()) {
                    cerr << "TinyObjReader: WARNING: \n";
                    cerr << reader.Warning() << endl;
                }

                size_t vert_offset = vertices.size();
                size_t norm_offset = normals.size();
                size_t uv_offset = uvs.size();
                size_t mat_offset = materials.size();

                auto const& attrib = reader.GetAttrib();
                auto const& mats = reader.GetMaterials();
                
                // fill textures
                for (auto const& mat : mats) {
                    
                }

                // fill vertices
                for (size_t i = 2; i < attrib.vertices.size(); i += 3) {
                    vertices.emplace_back(
                        attrib.vertices[i - 2],
                        attrib.vertices[i - 1],
                        attrib.vertices[i]
                    );
                }
                // fill normals
                for (size_t i = 2; i < attrib.normals.size(); i += 3) {
                    normals.emplace_back(
                        attrib.normals[i - 2],
                        attrib.normals[i - 1],
                        attrib.normals[i]
                    );
                }
                // fill uvs
                for (size_t i = 1; i < attrib.texcoords.size(); i += 2) {
                    uvs.emplace_back(
                        attrib.texcoords[i - 1],
                        attrib.texcoords[i]
                    );
                }
                
                int triangles_start = triangles.size();
                for (auto const& s : reader.GetShapes()) {
                    auto const& indices = s.mesh.indices;
                    for (size_t i = 0; i < s.mesh.material_ids.size(); ++i) {
                        int vals[]{
                            indices[3 * i + 0].vertex_index + vert_offset,
                            indices[3 * i + 1].vertex_index + vert_offset,
                            indices[3 * i + 2].vertex_index + vert_offset,

                            indices[3 * i + 0].normal_index + norm_offset,
                            indices[3 * i + 1].normal_index + norm_offset,
                            indices[3 * i + 2].normal_index + norm_offset,

                            indices[3 * i + 0].texcoord_index + uv_offset,
                            indices[3 * i + 1].texcoord_index + uv_offset,
                            indices[3 * i + 2].texcoord_index + uv_offset,

                            s.mesh.material_ids[i] + mat_offset
                        };

                        triangles.emplace_back(&vals);
                    }
                }

                newGeom.meshid = meshes.size();
                meshes.emplace_back(triangles_start, triangles.size());

                cout << "Loaded:\n"
                    << triangles.size() << " triangles\n"
                    << vertices.size() << " vertices\n"
                    << normals.size() << " normals\n"
                    << uvs.size() << " uvs\n"
                    << meshes.size() << " meshes\n";
            } else {
                cerr << "unknown object format" << endl;
                return -1;
            }
        }
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "material") {
            int mat_id = loadMaterial(tokens[1]);
            if (mat_id < 0) {
                return 1;
            }
            newGeom.materialid = mat_id;
            cout << "Connecting Geom " << objectid << " to Material " << tokens[1] << "..." << endl;
        } else {
            cerr << "unknown field: " << tokens[0] << endl;
            return -1;
        }
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    newGeom.transform = utilityCore::buildTransformationMatrix(
        newGeom.translation, newGeom.rotation, newGeom.scale);
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    // record lights
    Material const& mat = materials[newGeom.materialid];
    if (mat.emittance > 0) {
        Light light;
        light.color = mat.diffuse;
        light.intensity = mat.emittance / MAX_EMITTANCE;
        light.position = newGeom.translation;
        lights.emplace_back(light);
    }
    geoms.push_back(newGeom);
    return 1;
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}


static Material initMaterial(tinyobj::material_t const& tinyobj_mat) {
    Material mat;
    mat.diffuse = color_t(tinyobj_mat.diffuse[0], tinyobj_mat.diffuse[1], tinyobj_mat.diffuse[2]);
    mat.emittance = tinyobj_mat.emission[0];
    mat.ior = tinyobj_mat.ior;
    mat.hasReflective = tinyobj_mat.metallic;
    mat.hasRefractive = 1 - tinyobj_mat.dissolve;
    mat.roughness = tinyobj_mat.roughness;
    mat.specular.color = color_t(tinyobj_mat.specular[0], tinyobj_mat.specular[1], tinyobj_mat.specular[2]);
    mat.specular.exponent = tinyobj_mat.shininess;

    mat.textures.tex_idx = tinyobj_mat.diffuse_texname;

    return mat;
}

// standardize material using mtl format
// returns the material id on success or -1 on error
// NOTE: only loads 1 material per mtl file, the rest will be discarded
int Scene::loadMaterial(string mtl_file) {
    // maps mtl file to material index
    static unordered_map<string, int> s_loaded;
    
    if (s_loaded.count(mtl_file)) {
        return s_loaded[mtl_file];
    } else {
        ifstream fin(mtl_file);
        if (!fin) {
            cerr << "cannot find mtl file: " << mtl_file << endl;
            return -1;
        }

        map<string, int> mat_mp;
        vector<tinyobj::material_t> mats;
        string warn, err;

        tinyobj::LoadMtl(&mat_mp, &mats, &fin, &warn, &err);
        if (!err.empty()) {
            cerr << "Tiny obj loader: ERROR:\n" << err << endl;
            return -1;
        }
        if (!warn.empty()) {
            cerr << "Tiny obj loader: WARNING:\n" << err << endl;
        }

        if (mat_mp.size() > 1) {
            cerr << "WARNING: " << mat_mp.size()-1 << "materials discarded in " << mtl_file << endl;
        }

        materials.emplace_back(initMaterial(mats[0]));
        s_loaded[mtl_file] = materials.size() - 1;
        return materials.size() - 1;
    }
}