class Gecko_Codominance_Entity {
  constructor() {
    this.snow = 'S';
    this.lemonForst = 'L';
    this.supersnow = 'DS';
    this.superlemon = 'DL';
  }

};
class Gecko_Recessive_Entity {
  constructor() {
    this.tremp = 'T';
    this.bell = 'B';
    this.rainWater = 'R';
    this.giant = 'G';
  }
};
class Gecko_Breeding_Entity {
  constructor() {
    this.blackNight = false;
    this.Tangerine = false;
    this.cross = false;
  }
};
class Gecko_Dominant_Entity {
  constructor() {
    this.wy = 0;
    this.enigma = 0;
  }
}

module.exports = { Gecko_Breeding_Entity, Gecko_Codominance_Entity, Gecko_Dominant_Entity, Gecko_Recessive_Entity }


