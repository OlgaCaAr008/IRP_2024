// Simcenter STAR-CCM+ macro: testHyper.java
// Written by Simcenter STAR-CCM+ 18.04.009
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;
import star.realgas.*;
import star.material.*;
import star.coupledflow.*;
import star.vis.*;
import star.cadmodeler.*;
import star.flow.*;
import star.energy.*;
import star.meshing.*;

public class testHyper extends StarMacro {

  public void execute() {
    execute0();
    execute1();
  }

  private void execute0() {

    Simulation simulation_0 = 
      getActiveSimulation();

    Scene scene_0 = 
      simulation_0.getSceneManager().createScene("3D-CAD View");

    scene_0.initializeAndWait();

    CadModel cadModel_0 = 
      simulation_0.get(SolidModelManager.class).createSolidModel(scene_0);

    cadModel_0.resetSystemOptions();

    scene_0.openInteractive();

    scene_0.setAdvancedRenderingEnabled(false);

    SceneUpdate sceneUpdate_0 = 
      scene_0.getSceneUpdate();

    HardcopyProperties hardcopyProperties_0 = 
      sceneUpdate_0.getHardcopyProperties();

    hardcopyProperties_0.setCurrentResolutionWidth(25);

    hardcopyProperties_0.setCurrentResolutionHeight(25);

    hardcopyProperties_0.setCurrentResolutionWidth(880);

    hardcopyProperties_0.setCurrentResolutionHeight(347);

    scene_0.resetCamera();

    LabCoordinateSystem labCoordinateSystem_0 = 
      simulation_0.getCoordinateSystemManager().getLabCoordinateSystem();

    cadModel_0.getFeatureManager().create3DSketches_2("C:\\Users\\s425008\\Downloads\\smoothed_coordinates.csv", labCoordinateSystem_0, true, true);

    cadModel_0.update();

    scene_0.resetCamera();

    Sketch3D sketch3D_0 = 
      ((Sketch3D) cadModel_0.getFeature("Sketch3D 1"));

    ExtrusionMerge extrusionMerge_0 = 
      cadModel_0.getFeatureManager().createExtrusionMerge(sketch3D_0);

    extrusionMerge_0.setAutoPreview(true);

    cadModel_0.allowMakingPartDirty(false);

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.MAKE_SCENE_TRANSPARENT);

    extrusionMerge_0.setDirectionOption(2);

    extrusionMerge_0.setExtrudedBodyTypeOption(0);

    Units units_0 = 
      ((Units) simulation_0.getUnitsManager().getObject("m"));

    extrusionMerge_0.getDistance().setValueAndUnits(0.1, units_0);

    extrusionMerge_0.getDistanceAsymmetric().setValueAndUnits(0.1, units_0);

    extrusionMerge_0.getOffsetDistance().setValueAndUnits(0.1, units_0);

    extrusionMerge_0.setDistanceOption(0);

    extrusionMerge_0.setCoordinateSystemOption(1);

    Units units_1 = 
      ((Units) simulation_0.getUnitsManager().getObject("deg"));

    extrusionMerge_0.getDraftAngle().setValueAndUnits(10.0, units_1);

    extrusionMerge_0.setDraftOption(0);

    extrusionMerge_0.setImportedCoordinateSystem(labCoordinateSystem_0);

    extrusionMerge_0.getDirectionAxis().setCoordinateSystem(labCoordinateSystem_0);

    extrusionMerge_0.getDirectionAxis().setUnits0(units_0);

    extrusionMerge_0.getDirectionAxis().setUnits1(units_0);

    extrusionMerge_0.getDirectionAxis().setUnits2(units_0);

    extrusionMerge_0.getDirectionAxis().setDefinition("");

    extrusionMerge_0.getDirectionAxis().setValue(new DoubleVector(new double[] {0.0, 0.0, 1.0}));

    extrusionMerge_0.setFace(null);

    extrusionMerge_0.setBody(null);

    extrusionMerge_0.setFeatureInputType(0);

    extrusionMerge_0.setInputFeatureEdges(new NeoObjectVector(new Object[] {}));

    extrusionMerge_0.setSketch(sketch3D_0);

    extrusionMerge_0.setInteractingBodies(new NeoObjectVector(new Object[] {}));

    extrusionMerge_0.setInteractingBodiesBodyGroups(new NeoObjectVector(new Object[] {}));

    extrusionMerge_0.setInteractingBodiesCadFilters(new NeoObjectVector(new Object[] {}));

    extrusionMerge_0.setInteractingSelectedBodies(false);

    extrusionMerge_0.setPostOption(1);

    extrusionMerge_0.setExtrusionOption(0);

    extrusionMerge_0.setIsBodyGroupCreation(false);

    cadModel_0.getFeatureManager().markDependentNotUptodate(extrusionMerge_0);

    cadModel_0.allowMakingPartDirty(true);

    extrusionMerge_0.markFeatureForEdit();

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.USE_DISPLAYER_PROPERTY);

    cadModel_0.getFeatureManager().execute(extrusionMerge_0);

    CurrentView currentView_0 = 
      scene_0.getCurrentView();

    currentView_0.setInput(new DoubleVector(new double[] {34.465690591053466, 34.176172855434736, 0.03270858301502244}), new DoubleVector(new double[] {34.465690591053466, 34.176172855434736, 139.22010890306424}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {34.01449777252892, 33.34146614116433, 0.018577649103093563}), new DoubleVector(new double[] {34.01449777252892, 33.34146614116433, 168.43533187266902}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {33.46855446189713, 32.33147101649551, 0.0014788654090693854}), new DoubleVector(new double[] {33.46855446189713, 32.33147101649551, 203.7857516799481}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {33.15398712545976, 31.749521444086376, -0.008373323123180398}), new DoubleVector(new double[] {33.15398712545976, 31.749521444086376, 224.15432692240313}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {32.00864745221023, 29.630643048574754, -0.044245053251984245}), new DoubleVector(new double[] {32.00864745221023, 29.630643048574754, 298.31630946313186}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {49.049294841415495, 29.17008501102866, 0.09999916472827408}), new DoubleVector(new double[] {49.049294841415495, 29.17008501102866, 298.31630946313186}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {52.42979083858663, 26.51036234291976, -0.654010898650256}), new DoubleVector(new double[] {52.42979083858663, 26.51036234291976, 360.94173465594105}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    hardcopyProperties_0.setCurrentResolutionWidth(476);

    CanonicalSketchPlane canonicalSketchPlane_0 = 
      ((CanonicalSketchPlane) cadModel_0.getFeature("XY"));

    Sketch sketch_0 = 
      cadModel_0.getFeatureManager().createSketch(canonicalSketchPlane_0);

    sketch_0.setAutoPreview(true);

    cadModel_0.allowMakingPartDirty(false);

    cadModel_0.allowMakingPartDirty(false);

    cadModel_0.getFeatureManager().startSketchEdit(sketch_0);

    hardcopyProperties_0.setCurrentResolutionWidth(880);

    currentView_0.setInput(new DoubleVector(new double[] {52.42979083858663, 26.51036234291976, -0.654010898650256}), new DoubleVector(new double[] {52.42979083858663, 26.51036234291976, 360.94173465594105}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {34.33634468091233, 19.892851942659252, -49.191149596455205}), new DoubleVector(new double[] {34.33634468091233, 19.892851942659252, 397.1013089453314}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {8.887647891591453, 10.021643523090196, -108.43198091323228}), new DoubleVector(new double[] {8.887647891591453, 10.021643523090196, 439.57117080518157}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {62.075424793684185, 30.65246935753596, -107.74714911512024}), new DoubleVector(new double[] {62.075424793684185, 30.65246935753596, 350.8091584960756}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    sketch_0.createRectangle(new DoubleVector(new double[] {-107.0, -32.86}), new DoubleVector(new double[] {273.37, 118.69}));

    currentView_0.setInput(new DoubleVector(new double[] {62.075424793684185, 30.65246935753596, -107.74714911512024}), new DoubleVector(new double[] {62.075424793684185, 30.65246935753596, 350.8091584960756}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    sketch_0.setIsUptoDate(true);

    sketch_0.markFeatureForEdit();

    sketch_0.setIsBodyGroupCreation(false);

    cadModel_0.getFeatureManager().markDependentNotUptodate(sketch_0);

    cadModel_0.allowMakingPartDirty(true);

    sketch_0.markFeatureForEdit();

    cadModel_0.allowMakingPartDirty(true);

    cadModel_0.getFeatureManager().stopSketchEdit(sketch_0, true);

    cadModel_0.getFeatureManager().updateModelAfterFeatureEdited(sketch_0, null);

    ExtrusionMerge extrusionMerge_1 = 
      cadModel_0.getFeatureManager().createExtrusionMerge(sketch_0);

    extrusionMerge_1.setAutoPreview(true);

    cadModel_0.allowMakingPartDirty(false);

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.MAKE_SCENE_TRANSPARENT);

    extrusionMerge_1.setDirectionOption(0);

    extrusionMerge_1.setExtrudedBodyTypeOption(0);

    extrusionMerge_1.getDistance().setValueAndUnits(0.1, units_0);

    extrusionMerge_1.getDistanceAsymmetric().setValueAndUnits(0.1, units_0);

    extrusionMerge_1.getOffsetDistance().setValueAndUnits(0.1, units_0);

    extrusionMerge_1.setDistanceOption(0);

    extrusionMerge_1.setCoordinateSystemOption(0);

    extrusionMerge_1.getDraftAngle().setValueAndUnits(10.0, units_1);

    extrusionMerge_1.setDraftOption(0);

    extrusionMerge_1.setImportedCoordinateSystem(labCoordinateSystem_0);

    extrusionMerge_1.getDirectionAxis().setCoordinateSystem(labCoordinateSystem_0);

    extrusionMerge_1.getDirectionAxis().setUnits0(units_0);

    extrusionMerge_1.getDirectionAxis().setUnits1(units_0);

    extrusionMerge_1.getDirectionAxis().setUnits2(units_0);

    extrusionMerge_1.getDirectionAxis().setDefinition("");

    extrusionMerge_1.getDirectionAxis().setValue(new DoubleVector(new double[] {0.0, 0.0, 1.0}));

    extrusionMerge_1.setFace(null);

    extrusionMerge_1.setBody(null);

    extrusionMerge_1.setFeatureInputType(0);

    extrusionMerge_1.setInputFeatureEdges(new NeoObjectVector(new Object[] {}));

    extrusionMerge_1.setSketch(sketch_0);

    extrusionMerge_1.setInteractingBodies(new NeoObjectVector(new Object[] {}));

    extrusionMerge_1.setInteractingBodiesBodyGroups(new NeoObjectVector(new Object[] {}));

    extrusionMerge_1.setInteractingBodiesCadFilters(new NeoObjectVector(new Object[] {}));

    extrusionMerge_1.setInteractingSelectedBodies(false);

    extrusionMerge_1.setPostOption(0);

    extrusionMerge_1.setExtrusionOption(0);

    extrusionMerge_1.setIsBodyGroupCreation(false);

    cadModel_0.getFeatureManager().markDependentNotUptodate(extrusionMerge_1);

    cadModel_0.allowMakingPartDirty(true);

    extrusionMerge_1.markFeatureForEdit();

    cadModel_0.getFeatureManager().execute(extrusionMerge_1);

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.USE_DISPLAYER_PROPERTY);

    SubtractBodiesFeature subtractBodiesFeature_0 = 
      cadModel_0.getFeatureManager().createSubtractBodies();

    subtractBodiesFeature_0.setAutoPreview(true);

    cadModel_0.allowMakingPartDirty(false);

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.MAKE_SCENE_TRANSPARENT);

    currentView_0.setInput(new DoubleVector(new double[] {62.075424793684185, 30.65246935753596, -107.74714911512024}), new DoubleVector(new double[] {62.075424793684185, 30.65246935753596, 350.8091584960756}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    hardcopyProperties_0.setCurrentResolutionWidth(476);

    LineSketchPrimitive lineSketchPrimitive_0 = 
      ((LineSketchPrimitive) sketch_0.getSketchPrimitive("Line 1"));

    star.cadmodeler.Body cadmodelerBody_0 = 
      ((star.cadmodeler.Body) extrusionMerge_1.getBody(lineSketchPrimitive_0));

    subtractBodiesFeature_0.setTargetBodies(new NeoObjectVector(new Object[] {cadmodelerBody_0}));

    subtractBodiesFeature_0.setTargetBodyGroups(new NeoObjectVector(new Object[] {}));

    subtractBodiesFeature_0.setTargetCadFilters(new NeoObjectVector(new Object[] {}));

    SplineSketchPrimitive3D splineSketchPrimitive3D_0 = 
      ((SplineSketchPrimitive3D) sketch3D_0.getSketchPrimitive3D("Spline 1"));

    star.cadmodeler.Body cadmodelerBody_1 = 
      ((star.cadmodeler.Body) extrusionMerge_0.getBody(splineSketchPrimitive3D_0));

    subtractBodiesFeature_0.setToolBodies(new NeoObjectVector(new Object[] {cadmodelerBody_1}));

    subtractBodiesFeature_0.setToolBodyGroups(new NeoObjectVector(new Object[] {}));

    subtractBodiesFeature_0.setToolCadFilters(new NeoObjectVector(new Object[] {}));

    subtractBodiesFeature_0.setKeepToolBodies(false);

    subtractBodiesFeature_0.setImprint(false);

    subtractBodiesFeature_0.setImprintOption(0);

    subtractBodiesFeature_0.getTolerance().setValueAndUnits(1.0E-5, units_0);

    subtractBodiesFeature_0.setUseAutoMatch(true);

    subtractBodiesFeature_0.setTransferFaceNames(true);

    subtractBodiesFeature_0.setTransferBodyNames(false);

    subtractBodiesFeature_0.setIsBodyGroupCreation(false);

    cadModel_0.getFeatureManager().markDependentNotUptodate(subtractBodiesFeature_0);

    cadModel_0.allowMakingPartDirty(true);

    subtractBodiesFeature_0.markFeatureForEdit();

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.USE_DISPLAYER_PROPERTY);

    hardcopyProperties_0.setCurrentResolutionWidth(880);

    cadModel_0.getFeatureManager().execute(subtractBodiesFeature_0);

    currentView_0.setInput(new DoubleVector(new double[] {89.47464158304109, 24.421839287292748, -6.887035601056823}), new DoubleVector(new double[] {89.47464158304109, 24.421839287292748, 513.5732795670827}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 27.078424130672325, 0, 30.0);

    simulation_0.get(SolidModelManager.class).endEditCadModel(cadModel_0);

    simulation_0.getSceneManager().deleteScenes(new NeoObjectVector(new Object[] {scene_0}));

    cadModel_0.createParts(new NeoObjectVector(new Object[] {cadmodelerBody_0}), new NeoObjectVector(new Object[] {}), true, false, 1, false, false, 3, "SharpEdges", 30.0, 2, true, 1.0E-5, false);

    simulation_0.getSceneManager().createGeometryScene("Geometry Scene", "Outline", "Surface", 1, null);

    Scene scene_1 = 
      simulation_0.getSceneManager().getScene("Geometry Scene 1");

    scene_1.initializeAndWait();

    SceneUpdate sceneUpdate_1 = 
      scene_1.getSceneUpdate();

    HardcopyProperties hardcopyProperties_1 = 
      sceneUpdate_1.getHardcopyProperties();

    hardcopyProperties_1.setCurrentResolutionWidth(25);

    hardcopyProperties_1.setCurrentResolutionHeight(25);

    hardcopyProperties_1.setCurrentResolutionWidth(880);

    hardcopyProperties_1.setCurrentResolutionHeight(347);

    scene_1.resetCamera();

    scene_1.setTransparencyOverrideMode(SceneTransparencyOverride.MAKE_SCENE_TRANSPARENT);

    PartDisplayer partDisplayer_0 = 
      ((PartDisplayer) scene_1.getDisplayerManager().getObject("Outline 1"));

    SolidModelPart solidModelPart_0 = 
      ((SolidModelPart) simulation_0.get(SimulationPartManager.class).getPart("Body 2"));

    PartSurface partSurface_0 = 
      ((PartSurface) solidModelPart_0.getPartSurfaceManager().getPartSurface("Default"));

    partDisplayer_0.getHiddenParts().addObjects(partSurface_0);

    PartDisplayer partDisplayer_1 = 
      ((PartDisplayer) scene_1.getDisplayerManager().getObject("Surface 1"));

    partDisplayer_1.getHiddenParts().addObjects(partSurface_0);

    CurrentView currentView_1 = 
      scene_1.getCurrentView();

    currentView_1.setInput(new DoubleVector(new double[] {47.461135336764684, 42.23652390234585, -0.19800329871232236}), new DoubleVector(new double[] {47.461135336764684, 42.23652390234585, 41.40251003352476}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    currentView_1.setInput(new DoubleVector(new double[] {42.921322034308254, 39.6788826051873, 6.266724028591852E-9}), new DoubleVector(new double[] {42.921322034308254, 39.6788826051873, 41.40251003352476}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    currentView_1.setInput(new DoubleVector(new double[] {18.722691230410998, 45.97142887806852, -1.6378661749105845}), new DoubleVector(new double[] {18.722691230410998, 45.97142887806852, 230.19453234893874}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    solidModelPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] {1, 8}), "object");

    currentView_1.setInput(new DoubleVector(new double[] {57.79141525143678, 48.093327000593334, -21.80967924389489}), new DoubleVector(new double[] {57.79141525143678, 48.093327000593334, 336.98140527304304}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    solidModelPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] {2}), "symm");

    solidModelPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] {3}), "symm2");

    currentView_1.setInput(new DoubleVector(new double[] {-1.2701600100798238, 76.81872951414914, -37.083295064345066}), new DoubleVector(new double[] {-1.2701600100798238, 76.81872951414914, 221.09350000196957}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    currentView_1.setInput(new DoubleVector(new double[] {-67.116138637873, 103.70650468855204, -10.662283631409451}), new DoubleVector(new double[] {-67.116138637873, 103.70650468855204, 77.09053669377617}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    solidModelPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] {4}), "inlet");

    solidModelPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] {5}), "top");

    currentView_1.setInput(new DoubleVector(new double[] {-63.02326351685692, 114.27331382291355, -1.0716213024160481}), new DoubleVector(new double[] {-63.02326351685692, 114.27331382291355, 181.77545299477418}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    currentView_1.setInput(new DoubleVector(new double[] {-21.475282523099594, 67.11074080297284, 2.4669134290888906E-8}), new DoubleVector(new double[] {-21.475282523099594, 67.11074080297284, 181.77545299477418}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    currentView_1.setInput(new DoubleVector(new double[] {15.58102484971097, 1.981473299245181, 2.4669134290888906E-8}), new DoubleVector(new double[] {15.58102484971097, 1.981473299245181, 181.77545299477418}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    currentView_1.setInput(new DoubleVector(new double[] {13.545735240361143, 4.896850208563663, -1.4187912326529784}), new DoubleVector(new double[] {13.545735240361143, 4.896850208563663, 219.9482981158754}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    currentView_1.setInput(new DoubleVector(new double[] {7.870450596569594, 17.429770463603337, -2.670992143949377}), new DoubleVector(new double[] {7.870450596569594, 17.429770463603337, 322.02630327423356}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    currentView_1.setInput(new DoubleVector(new double[] {83.46455178705868, 43.29091034456015, 1.1894832141479128E-7}), new DoubleVector(new double[] {83.46455178705868, 43.29091034456015, 322.02630327423356}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72466229059947, 0, 30.0);

    solidModelPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] {6}), "outlet");

    partDisplayer_0.getVisibleParts().addParts(partSurface_0);

    partDisplayer_0.getHiddenParts().addParts();

    partDisplayer_1.getVisibleParts().addParts(partSurface_0);

    partDisplayer_1.getHiddenParts().addParts();

    scene_1.setTransparencyOverrideMode(SceneTransparencyOverride.USE_DISPLAYER_PROPERTY);

    partSurface_0.setPresentationName("bot");

    Region region_0 = 
      simulation_0.getRegionManager().createEmptyRegion(null);

    region_0.setPresentationName("Region");

    Boundary boundary_0 = 
      region_0.getBoundaryManager().getBoundary("Default");

    region_0.getBoundaryManager().removeBoundaries(new NeoObjectVector(new Object[] {boundary_0}));

    simulation_0.getRegionManager().newRegionsFromParts(new NeoObjectVector(new Object[] {solidModelPart_0}), "OneRegion", region_0, "OneBoundaryPerPartSurface", null, RegionManager.CreateInterfaceMode.BOUNDARY, "OneEdgeBoundaryPerPart", null);

    Boundary boundary_1 = 
      region_0.getBoundaryManager().getBoundary("Body 2.bot");

    FreeStreamBoundary freeStreamBoundary_0 = 
      ((FreeStreamBoundary) simulation_0.get(ConditionTypeManager.class).get(FreeStreamBoundary.class));

    boundary_1.setBoundaryType(freeStreamBoundary_0);

    Boundary boundary_2 = 
      region_0.getBoundaryManager().getBoundary("Body 2.inlet");

    boundary_2.setBoundaryType(freeStreamBoundary_0);

    Boundary boundary_3 = 
      region_0.getBoundaryManager().getBoundary("Body 2.outlet");

    boundary_3.setBoundaryType(freeStreamBoundary_0);

    Boundary boundary_4 = 
      region_0.getBoundaryManager().getBoundary("Body 2.symm");

    SymmetryBoundary symmetryBoundary_0 = 
      ((SymmetryBoundary) simulation_0.get(ConditionTypeManager.class).get(SymmetryBoundary.class));

    boundary_4.setBoundaryType(symmetryBoundary_0);

    Boundary boundary_5 = 
      region_0.getBoundaryManager().getBoundary("Body 2.symm2");

    boundary_5.setBoundaryType(symmetryBoundary_0);

    Boundary boundary_6 = 
      region_0.getBoundaryManager().getBoundary("Body 2.top");

    boundary_6.setBoundaryType(freeStreamBoundary_0);

    PrepareFor2dOperation prepareFor2dOperation_0 = 
      (PrepareFor2dOperation) simulation_0.get(MeshOperationManager.class).createPrepareFor2dOperation(new NeoObjectVector(new Object[] {solidModelPart_0}));

    prepareFor2dOperation_0.execute();

    AutoMeshOperation2d autoMeshOperation2d_0 = 
      simulation_0.get(MeshOperationManager.class).createAutoMeshOperation2d(new StringVector(new String[] {"star.twodmesher.DualAutoMesher2d", "star.prismmesher.PrismAutoMesher"}), new NeoObjectVector(new Object[] {solidModelPart_0}));

    MeshPipelineController meshPipelineController_0 = 
      simulation_0.get(MeshPipelineController.class);

    meshPipelineController_0.generateVolumeMesh();

    simulation_0.getSceneManager().createGeometryScene("Mesh Scene", "Outline", "Mesh", 3, null);

    Scene scene_2 = 
      simulation_0.getSceneManager().getScene("Mesh Scene 1");

    scene_2.initializeAndWait();

    SceneUpdate sceneUpdate_2 = 
      scene_2.getSceneUpdate();

    HardcopyProperties hardcopyProperties_2 = 
      sceneUpdate_2.getHardcopyProperties();

    hardcopyProperties_2.setCurrentResolutionWidth(25);

    hardcopyProperties_2.setCurrentResolutionHeight(25);

    hardcopyProperties_1.setCurrentResolutionWidth(882);

    hardcopyProperties_1.setCurrentResolutionHeight(348);

    hardcopyProperties_2.setCurrentResolutionWidth(880);

    hardcopyProperties_2.setCurrentResolutionHeight(347);

    scene_2.resetCamera();

    CurrentView currentView_2 = 
      scene_2.getCurrentView();

    currentView_2.setInput(new DoubleVector(new double[] {90.88104226449694, 45.89752177590218, -0.34157727650392644}), new DoubleVector(new double[] {90.88104226449694, 45.89752177590218, 576.6355954113111}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72465618483767, 0, 30.0);

    currentView_2.setInput(new DoubleVector(new double[] {84.2019770216103, 48.01255910280044, -1.0978569388560118}), new DoubleVector(new double[] {84.2019770216103, 48.01255910280044, 467.07483192853215}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72465618483767, 0, 30.0);

    currentView_2.setInput(new DoubleVector(new double[] {80.88381742319318, 48.769965098091305, -1.252137538317072}), new DoubleVector(new double[] {80.88381742319318, 48.769965098091305, 420.36734874925463}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 204.72465618483767, 0, 30.0);

    PhysicsContinuum physicsContinuum_0 = 
      ((PhysicsContinuum) simulation_0.getContinuumManager().getContinuum("Physics 1"));

    physicsContinuum_0.enable(SteadyModel.class);

    physicsContinuum_0.enable(SingleComponentGasModel.class);

    physicsContinuum_0.enable(CoupledFlowModel.class);

    physicsContinuum_0.enable(RealGasModel.class);

    physicsContinuum_0.enable(CoupledEnergyModel.class);

    physicsContinuum_0.enable(EquilibriumAirEosModel.class);

    physicsContinuum_0.enable(LaminarModel.class);
  }

  private void execute1() {

    Simulation simulation_0 = 
      getActiveSimulation();

    PhysicsContinuum physicsContinuum_0 = 
      ((PhysicsContinuum) simulation_0.getContinuumManager().getContinuum("Physics 1"));

    CoupledFlowModel coupledFlowModel_0 = 
      physicsContinuum_0.getModelManager().getModel(CoupledFlowModel.class);

    coupledFlowModel_0.getUpwindOption().setSelected(FlowUpwindOption.Type.MUSCL_3RD_ORDER);

    coupledFlowModel_0.setPositivityRate(0.05);

    coupledFlowModel_0.setUnsteadyPreconditioningEnabled(false);

    coupledFlowModel_0.getCoupledInviscidFluxOption().setSelected(CoupledInviscidFluxOption.Type.AUSM_SCHEME);

    Units units_2 = 
      ((Units) simulation_0.getUnitsManager().getObject("K"));

    physicsContinuum_0.getReferenceValues().get(MaximumAllowableTemperature.class).setValueAndUnits(10000.0, units_2);

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("Region");

    Boundary boundary_1 = 
      region_0.getBoundaryManager().getBoundary("Body 2.bot");

    MachNumberProfile machNumberProfile_0 = 
      boundary_1.getValues().get(MachNumberProfile.class);

    Units units_3 = 
      ((Units) simulation_0.getUnitsManager().getObject(""));

    machNumberProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(12.0, units_3);

    StaticTemperatureProfile staticTemperatureProfile_0 = 
      boundary_1.getValues().get(StaticTemperatureProfile.class);

    staticTemperatureProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(241.0, units_2);

    Boundary boundary_2 = 
      region_0.getBoundaryManager().getBoundary("Body 2.inlet");

    MachNumberProfile machNumberProfile_1 = 
      boundary_2.getValues().get(MachNumberProfile.class);

    machNumberProfile_1.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(12.0, units_3);

    StaticTemperatureProfile staticTemperatureProfile_1 = 
      boundary_2.getValues().get(StaticTemperatureProfile.class);

    staticTemperatureProfile_1.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(241.0, units_2);

    Boundary boundary_3 = 
      region_0.getBoundaryManager().getBoundary("Body 2.outlet");

    MachNumberProfile machNumberProfile_2 = 
      boundary_3.getValues().get(MachNumberProfile.class);

    machNumberProfile_2.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(12.0, units_3);

    StaticTemperatureProfile staticTemperatureProfile_2 = 
      boundary_3.getValues().get(StaticTemperatureProfile.class);

    staticTemperatureProfile_2.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(241.0, units_2);

    Boundary boundary_6 = 
      region_0.getBoundaryManager().getBoundary("Body 2.top");

    MachNumberProfile machNumberProfile_3 = 
      boundary_6.getValues().get(MachNumberProfile.class);

    machNumberProfile_3.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(241.0, units_3);

    StaticTemperatureProfile staticTemperatureProfile_3 = 
      boundary_6.getValues().get(StaticTemperatureProfile.class);

    staticTemperatureProfile_3.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(241.0, units_2);

    machNumberProfile_3.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(12.0, units_3);

    StaticTemperatureProfile staticTemperatureProfile_4 = 
      physicsContinuum_0.getInitialConditions().get(StaticTemperatureProfile.class);

    staticTemperatureProfile_4.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(241.0, units_2);

    VelocityProfile velocityProfile_0 = 
      physicsContinuum_0.getInitialConditions().get(VelocityProfile.class);

    Units units_4 = 
      ((Units) simulation_0.getUnitsManager().getObject("m/s"));

    velocityProfile_0.getMethod(ConstantVectorProfileMethod.class).getQuantity().setComponentsAndUnits(3736.0, 0.0, 0.0, units_4);

    CoupledImplicitSolver coupledImplicitSolver_0 = 
      ((CoupledImplicitSolver) simulation_0.getSolverManager().getSolver(CoupledImplicitSolver.class));

    AMGLinearSolver aMGLinearSolver_0 = 
      coupledImplicitSolver_0.getAMGLinearSolver();

    aMGLinearSolver_0.setConvergeTol(0.001);

    aMGLinearSolver_0.getSmootherOption().setSelected(AMGSmootherOption.Type.ILU);

    coupledImplicitSolver_0.getExpertInitManager().getExpertInitOption().setSelected(ExpertInitOption.Type.GRID_SEQ_METHOD);

    GridSequencingInit gridSequencingInit_0 = 
      ((GridSequencingInit) coupledImplicitSolver_0.getExpertInitManager().getInit());

    gridSequencingInit_0.setGSCfl(2.5);
  }
}
