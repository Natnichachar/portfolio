import React, { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowLeft,
  Compass,
  Fish,
  Search,
  Sparkles,
  Turtle,
  Waves,
  Wind,
} from "lucide-react";

const rooms = [
  {
    id: 1,
    title: "The Ocean Realm",
    caption: "Understanding the vast blue world and its dynamics.",
    themeClass: "theme-ocean",
    icon: Waves,
    facts: [
      {
        title: "The Seven Ancient Seas",
        body: "In ancient Greece, the Seven Seas referred to regional bodies of water such as the Indian Ocean, Black Sea, Caspian Sea, Adriatic Sea, Persian Gulf, Mediterranean Sea, and Red Sea.",
        image: "/images/TheSevenAncientSeas.webp",
      },
      {
        title: "The Origin of Wind",
        body: "Wind forms as air moves from high-pressure regions to low-pressure regions, helping create surface wind waves.",
        image: "/images/TheOriginOfWind.png",
      },
      {
        title: "Tsunami History in Portugal",
        body: "A devastating tsunami struck Portugal in 1755 after the Great Lisbon earthquake, becoming one of Europe’s most significant natural hazards.",
        image: "/images/TsunamiHistoryInPortugal.jpg",
      },
      {
        title: "Ocean Currents and Ecosystems",
        body: "Ocean currents influence migration routes and the distribution of ecosystems, including where coral reefs can thrive.",
        image: "https://images.unsplash.com/photo-1544551763-77ef2d0cfc6c?auto=format&fit=crop&w=1400&q=80",
      },
      {
        title: "The Global Conveyor Belt",
        body: "Thermohaline circulation moves water around the planet using differences in temperature and salinity, reaching deep ocean layers.",
        image: "/images/TheGlobalConveyorBelt.jpg",
      },
    ],
  },
  {
    id: 2,
    title: "The Drifting Life",
    caption: "Exploring the microscopic wonders that fuel the sea.",
    themeClass: "theme-drifting",
    icon: Compass,
    facts: [
      {
        title: "Defining the Drifters",
        body: "Plankton are organisms that cannot swim against currents. The term describes a drifting lifestyle rather than a specific size.",
        image: "/images/DefiningTheDrifters.jpg",
      },
      {
        title: "The Ocean’s Lungs",
        body: "Phytoplankton use photosynthesis to produce oxygen and form the foundation of aquatic food webs.",
        image: "/images/TheOcean'sLungs.jpg",
      },
      {
        title: "Star Sand of Okinawa",
        body: "Some Okinawan star sand is made of the discarded shells of tiny zooplankton called foraminiferans.",
        image: "/images/StarSandofOkinawa.jpg",
      },
      {
        title: "Architecture Inspired by Radiolaria",
        body: "The Monumental Gate at the 1900 Universelle Exhibition in Paris was inspired by radiolaria skeletal structures.",
        image: "/images/ArchitectureInspiredByRadiolaria.jpg",
      },
      {
        title: "Plankton’s Economic Impact",
        body: "Plankton support the marine food web, directly affecting fish populations and global fisheries.",
        image: "/images/Plankton’sEconomicImpact.png",
      },
    ],
  },
  {
    id: 3,
    title: "Invertebrate Oddities",
    caption: "Celebrating the diverse lives of animals without backbones.",
    themeClass: "theme-oddities",
    icon: Fish,
    facts: [
      {
        title: "The Invertebrate Majority",
        body: "Invertebrates make up more than 90% of animal species, and many of them inhabit the oceans.",
        image: "https://images.unsplash.com/photo-1546026423-cc4642628d2b?auto=format&fit=crop&w=1400&q=80",
      },
      {
        title: "The Ambush Predator",
        body: "The Bobbit worm hides in sand and can rapidly strike prey with powerful jaws.",
        image: "/images/TheAmbushPredator.jpg",
      },
      {
        title: "The Largest Terrestrial Arthropod",
        body: "The coconut crab is the largest terrestrial arthropod and is actually a type of hermit crab.",
        image: "/images/TheLargestTerrestrialArthropod.jpg",
      },
      {
        title: "The Venomous Blue Ring",
        body: "Blue-ringed octopuses display vivid rings when threatened and carry highly potent venom.",
        image: "/images/TheVenomousBlueRing.jpg",
      },
      {
        title: "Master Mimics",
        body: "The mimic octopus can imitate other sea creatures by changing its shape and color.",
        image: "/images/MasterMImics.jpg",
      },
    ],
  },
  {
    id: 4,
    title: "Giants and Survivors",
    caption: "From ancient reptiles to the masters of the deep.",
    themeClass: "theme-giants",
    icon: Turtle,
    facts: [
      {
        title: "The Largest Captive Crocodile",
        body: "Lolong, a saltwater crocodile from the Philippines, was the largest crocodile ever held in captivity.",
        image: "/images/TheLargestCaptiveCrocodile.jpg",
      },
      {
        title: "The Secret of Shark Buoyancy",
        body: "Sharks stay buoyant using a large, oily liver rich in squalene instead of a swim bladder.",
        image: "https://images.unsplash.com/photo-1560275619-4662e36fa65c?auto=format&fit=crop&w=1400&q=80",
      },
      {
        title: "Clownfish Gender Dynamics",
        body: "Clownfish are born male, and the largest male can transform into a female if needed.",
        image: "/images/ClownfishGenderDynamics.jpg",
      },
      {
        title: "Bony Fish Protection",
        body: "Bony fish have an operculum, a protective flap over the gills that helps them breathe without constant swimming.",
        image: "/images/BonyFishProtection.jpg",
      },
      {
        title: "Reptilian Adaptation",
        body: "Marine reptiles have dry, scaly skin to reduce water loss and still rely on lungs for breathing.",
        image: "/images/ReptilianAdaptation.jpg",
      },
    ],
  },
];

const quickTags = [
  { label: "currents", icon: Wind },
  { label: "plankton", icon: Fish },
  { label: "invertebrates", icon: Compass },
  { label: "octopus", icon: Sparkles },
];

function Chip({ children, className = "" }) {
  return <div className={`chip ${className}`}>{children}</div>;
}

function RoomCard({ room, onOpen }) {
  const Icon = room.icon;

  return (
    <motion.button
      whileHover={{ y: -6 }}
      whileTap={{ scale: 0.985 }}
      className="room-card"
      onClick={onOpen}
      type="button"
    >
      <div className={`room-card__hero ${room.themeClass}`}>
        <div className="room-card__top">
          <Chip>Room {room.id}</Chip>
          <Icon size={28} />
        </div>
        <h3>{room.title}</h3>
        <p>{room.caption}</p>
      </div>

      <div className="room-card__body">
        <div className="topic-preview-list">
          {room.facts.slice(0, 3).map((fact) => (
            <div key={fact.title} className="topic-preview-item">
              {fact.title}
            </div>
          ))}
        </div>

        <button className="primary-button" type="button">
          Explore room
        </button>
      </div>
    </motion.button>
  );
}

function FactPanel({ fact, index, onSelect }) {
  return (
    <motion.button
      type="button"
      onClick={() => onSelect(fact)}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="fact-panel"
    >
      <div className="fact-panel__index">{index + 1}</div>
      <div className="fact-panel__title">{fact.title}</div>
    </motion.button>
  );
}

function TopicDetail({ fact, onBack }) {
  return (
    <motion.div
      key={fact.title}
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      className="page-section"
    >
      <button className="ghost-button" onClick={onBack} type="button">
        <ArrowLeft size={18} />
        Back to topics
      </button>

      <div className="detail-card">
        <div className="detail-image-wrap">
          <img src={fact.image} alt={fact.title} className="detail-image" />
        </div>
        <div className="detail-content">
          <h2>{fact.title}</h2>
          <p>{fact.body}</p>
        </div>
      </div>
    </motion.div>
  );
}

export default function App() {
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [selectedFact, setSelectedFact] = useState(null);
  const [query, setQuery] = useState("");

  const filteredRooms = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return rooms;

    return rooms.filter((room) => {
      const haystack = [
        room.title,
        room.caption,
        ...room.facts.flatMap((fact) => [fact.title, fact.body]),
      ]
        .join(" ")
        .toLowerCase();

      return haystack.includes(q);
    });
  }, [query]);

  return (
    <div className="app-shell">
      <div className="container">
        {!selectedRoom && (
          <motion.section
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            className="hero-card"
          >
            <div className="hero-grid">
              <div>
                <Chip className="hero-chip">Interactive ocean exhibit</Chip>
                <h1>Explore realms of rooms</h1>
                <p className="hero-copy">
                  A web-based learning journey inspired by an underwater exhibit. Enter themed
                  rooms, discover standout marine facts, and browse each realm like a digital museum.
                </p>
              </div>

              <div className="search-panel">
                <div className="search-input-wrap">
                  <Search size={16} />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search oceans, plankton, octopus..."
                    className="search-input"
                  />
                </div>

                <div className="tag-row">
                  {quickTags.map((tag) => {
                    const Icon = tag.icon;
                    return (
                      <button
                        key={tag.label}
                        className="tag-button"
                        onClick={() => setQuery(tag.label)}
                        type="button"
                      >
                        <Icon size={16} />
                        {tag.label}
                      </button>
                    );
                  })}

                  {query && (
                    <button className="ghost-button compact" onClick={() => setQuery("")} type="button">
                      Clear
                    </button>
                  )}
                </div>
              </div>
            </div>
          </motion.section>
        )}

        <AnimatePresence mode="wait">
          {!selectedRoom ? (
            <motion.section
              key="grid"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="page-section"
            >
              <div className="section-header">
                <div>
                  <h2>Choose a room</h2>
                  <p>Each room groups a distinct marine theme.</p>
                </div>
                <Chip className="count-chip">
                  <Sparkles size={16} />
                  {filteredRooms.length} room{filteredRooms.length === 1 ? "" : "s"} found
                </Chip>
              </div>

              {filteredRooms.length > 0 ? (
                <div className="room-grid">
                  {filteredRooms.map((room) => (
                    <RoomCard
                      key={room.id}
                      room={room}
                      onOpen={() => {
                        setSelectedRoom(room);
                        setSelectedFact(null);
                      }}
                    />
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <h3>No rooms matched your search.</h3>
                  <p>Try searching for currents, plankton, invertebrates, or octopus.</p>
                  <button className="primary-button" onClick={() => setQuery("")} type="button">
                    Reset search
                  </button>
                </div>
              )}
            </motion.section>
          ) : selectedFact ? (
            <TopicDetail fact={selectedFact} onBack={() => setSelectedFact(null)} />
          ) : (
            <motion.section
              key="topics"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="page-section"
            >
              <div className="top-actions">
                <button
                  className="ghost-button"
                  onClick={() => {
                    setSelectedRoom(null);
                    setSelectedFact(null);
                  }}
                  type="button"
                >
                  <ArrowLeft size={18} />
                  Back to all rooms
                </button>
              </div>

              <div className={`topics-card ${selectedRoom.themeClass}`}>
                <div className="topics-card__inner">
                  <div className="section-header">
                    <div>
                      <Chip>Room {selectedRoom.id}</Chip>
                      <h2>{selectedRoom.title}</h2>
                      <p>{selectedRoom.caption}</p>
                      <p className="subtle-note">Select a topic to open its photo and details.</p>
                    </div>
                    <div className="topics-count">{selectedRoom.facts.length} topics</div>
                  </div>

                  <div className="topics-grid">
                    {selectedRoom.facts.map((fact, index) => (
                      <FactPanel
                        key={fact.title}
                        fact={fact}
                        index={index}
                        onSelect={setSelectedFact}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
