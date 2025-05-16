import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

class ActivityRecommender:
    def __init__(self):
        # Initialize data structures
        self.users = {}
        self.user_pins = defaultdict(set)
        self.user_likes = defaultdict(set)
        self.user_dislikes = defaultdict(set)
        self.user_ratings = defaultdict(dict)
        
        # Color theme
        self.theme = {
            'bg': '#121212',  # Dark background
            'fg': '#e0e0e0',  # Light text
            'accent': '#9abf7f',  # Sage green
            'card_bg': '#1e1e1e',  # Slightly lighter dark
            'button_bg': '#2e2e2e',
            'like': '#4CAF50',  # Green
            'dislike': '#F44336',  # Red
            'pin': '#2196F3'  # Blue
        }
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Activity Recommender ")
        self.root.geometry("1100x800")
        self.root.configure(bg=self.theme['bg'])
        self.current_user = None
        self.setup_login_screen()
        self.root.mainloop()

    # Data Loading and Preparation
    def load_data(self):
        """Load and prepare the activity dataset"""
        try:
            self.activities = pd.read_csv("Activity Prediction\Activity_DataSet.csv")
            if len(self.activities) == 0:
                raise ValueError("Dataset is empty")
            
            # Ensure participants are on 1-5 scale
            if 'Participants' in self.activities.columns:
                self.activities['Participants'] = self.convert_to_scale(self.activities['Participants'], 1, 5)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            # Create default dataset
            self.activities = pd.DataFrame({
                'Activity': ['Hiking', 'Movie Night', 'Cooking Class', 'Museum Visit'],
                'Type': ['Outdoor', 'Indoor', 'Indoor', 'Cultural'],
                'Price': [0.3, 0.7, 0.5, 0.6],
                'Participants': [4, 2, 3, 3],  # 1-5 scale
                'Description': [
                    'Enjoy nature on a scenic trail',
                    'Watch the latest blockbuster at home',
                    'Learn to make pasta from scratch',
                    'Explore ancient artifacts and history'
                ]
            })
        
        # Ensure required columns
        self.activities['id'] = self.activities.index
        if 'Description' not in self.activities.columns:
            self.activities['Description'] = "No description available"
        else:
            self.activities['Description'] = self.activities['Description'].fillna("No description available")

    def convert_to_scale(self, series, min_val, max_val):
        """Convert values to a specified scale"""
        current_min = series.min()
        current_max = series.max()
        if current_max - current_min == 0:
            return np.full(len(series), min_val)
        return ((series - current_min) / (current_max - current_min)) * (max_val - min_val) + min_val

    def prepare_data(self):
        """Prepare data for recommendation engine"""
        # Normalize price (participants already 1-5)
        if 'Price' in self.activities.columns:
            scaler = MinMaxScaler()
            self.activities['Price'] = scaler.fit_transform(self.activities[['Price']])
        
        # One-hot encode activity types
        if 'Type' in self.activities.columns:
            self.category_dummies = pd.get_dummies(self.activities['Type'])
            self.content_features = pd.concat([self.activities[['Price']], self.category_dummies], axis=1)
        else:
            self.content_features = self.activities[['Price']]
        
        # Compute similarity matrix
        self.content_similarity = cosine_similarity(self.content_features)

    # User Management
    def create_user(self, username):
        """Create new user account"""
        if username in self.users:
            return False
        self.users[username] = {
            'user_id': len(self.users),
            'preferences': {
                'activity_types': set(),
                'price_range': (0, 1),
                'participants_range': (1, 5)  # 1-5 scale
            }
        }
        return True

    def pin_activity(self, activity_id):
        """Pin activity for current user"""
        self.user_pins[self.current_user].add(activity_id)
        messagebox.showinfo("Pinned", "Activity added to your pins!")

    def like_activity(self, activity_id):
        """Like activity for current user"""
        self.user_likes[self.current_user].add(activity_id)
        if activity_id in self.user_dislikes[self.current_user]:
            self.user_dislikes[self.current_user].remove(activity_id)
        messagebox.showinfo("Liked", "Activity added to your likes!")

    def dislike_activity(self, activity_id):
        """Dislike activity for current user"""
        self.user_dislikes[self.current_user].add(activity_id)
        if activity_id in self.user_likes[self.current_user]:
            self.user_likes[self.current_user].remove(activity_id)
        messagebox.showinfo("Disliked", "We'll avoid similar activities")

    def rate_activity(self, activity_id, rating):
        """Rate activity (1-5) for current user"""
        self.user_ratings[self.current_user][activity_id] = rating
        messagebox.showinfo("Rating Saved", f"Thanks for your {rating}-star rating!")

    # Recommendation Engine
    def get_recommendations(self, top_n=5):
        """Generate personalized recommendations"""
        try:
            user_data = self.users.get(self.current_user)
            if not user_data:
                return []
            
            preferences = user_data['preferences']
            scores = np.zeros(len(self.activities))
            
            # Content-based filtering
            for activity_id, rating in self.user_ratings[self.current_user].items():
                similarity_vector = self.content_similarity[int(activity_id)]
                scores += similarity_vector * rating
            
            # Boost liked and pinned activities
            for activity_id in self.user_likes[self.current_user]:
                scores[activity_id] += 0.5
            for activity_id in self.user_pins[self.current_user]:
                scores[activity_id] += 0.3
            # Penalize disliked activities
            for activity_id in self.user_dislikes[self.current_user]:
                scores[activity_id] -= 1.0
            
            # Filter by preferences
            for idx, row in self.activities.iterrows():
                # Activity type filter
                if preferences['activity_types'] and row['Type'] not in preferences['activity_types']:
                    scores[idx] = -10
                    continue
                
                # Price range filter
                if not (preferences['price_range'][0] <= row['Price'] <= preferences['price_range'][1]):
                    scores[idx] = -10
                    continue
                    
                # Participants filter (1-5 scale)
                if not (preferences['participants_range'][0] <= row['Participants'] <= preferences['participants_range'][1]):
                    scores[idx] = -10
                    continue
            
            # Exclude already interacted activities
            interacted = set(self.user_pins[self.current_user]).union(
                set(self.user_likes[self.current_user]),
                set(self.user_dislikes[self.current_user]),
                set(self.user_ratings[self.current_user].keys())
            )
            for activity_id in interacted:
                scores[int(activity_id)] = -10
            
            # Get top recommendations
            top_indices = np.argsort(scores)[::-1][:top_n]
            return self.activities.iloc[top_indices].to_dict('records')
        
        except Exception as e:
            messagebox.showerror("Error", f"Recommendation error: {str(e)}")
            return self.activities.sample(min(top_n, len(self.activities))).to_dict('records')

    # GUI Components
    def create_scrollable_frame(self, parent):
        """Create a scrollable frame with canvas and scrollbar"""
        container = tk.Frame(parent, bg=self.theme['bg'])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(container, bg=self.theme['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.theme['bg'])
        
        # Configure canvas scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        # Create window in canvas for scrollable frame
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return scrollable_frame

    def setup_login_screen(self):
        """Create login/registration screen"""
        self.clear_window()
        
        # Main container
        container = tk.Frame(self.root, bg=self.theme['bg'])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable frame
        main_frame = self.create_scrollable_frame(container)
        
        # Title
        title_frame = tk.Frame(main_frame, bg=self.theme['bg'])
        title_frame.pack(pady=40)
        tk.Label(title_frame, text="ACTIVIGO✅", 
                font=("Helvetica", 24, "bold"), bg=self.theme['bg'], fg=self.theme['accent']).pack()
        
        # Login form
        form_frame = tk.Frame(main_frame, bg=self.theme['bg'])
        form_frame.pack(pady=20)
        
        tk.Label(form_frame, text="Username:", font=("Helvetica", 12), 
                bg=self.theme['bg'], fg=self.theme['fg']).grid(row=0, column=0, padx=5, pady=5)
        self.username_entry = tk.Entry(form_frame, font=("Helvetica", 12), 
                                    bg=self.theme['card_bg'], fg=self.theme['fg'], insertbackground='white')
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Buttons
        btn_frame = tk.Frame(main_frame, bg=self.theme['bg'])
        btn_frame.pack(pady=20)
        
        login_btn = tk.Button(btn_frame, text="Login", command=self.handle_login,
                            font=("Helvetica", 12), width=15,
                            bg=self.theme['accent'], fg='black', activebackground='#b8d8a3')
        login_btn.pack(side=tk.LEFT, padx=10)
        
        register_btn = tk.Button(btn_frame, text="Register", command=self.handle_register,
                            font=("Helvetica", 12), width=15,
                            bg=self.theme['button_bg'], fg=self.theme['fg'],
                            activebackground='#3e3e3e')
        register_btn.pack(side=tk.LEFT, padx=10)

    def handle_login(self):
        """Handle login attempt"""
        username = self.username_entry.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
            
        if username in self.users:
            self.current_user = username
            self.setup_home_screen()
        else:
            messagebox.showerror("Error", "Username not found")

    def handle_register(self):
        """Handle new registration"""
        username = self.username_entry.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return
            
        if self.create_user(username):
            self.current_user = username
            self.setup_preferences_screen()
        else:
            messagebox.showerror("Error", "Username already exists")

    def setup_preferences_screen(self):
        """Screen for setting user preferences"""
        self.clear_window()
        
        # Main container
        container = tk.Frame(self.root, bg=self.theme['bg'])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable frame
        main_frame = self.create_scrollable_frame(container)
        
        # Title
        title_frame = tk.Frame(main_frame, bg=self.theme['bg'])
        title_frame.pack(pady=20)
        tk.Label(title_frame, text="Set Your Preferences", 
                font=("Helvetica", 20, "bold"), bg=self.theme['bg'], fg=self.theme['accent']).pack()
        
        # Preferences frame
        pref_frame = tk.LabelFrame(main_frame, text="Activity Preferences", padx=10, pady=10,
                                bg=self.theme['card_bg'], fg=self.theme['accent'],
                                font=("Helvetica", 12, "bold"))
        pref_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Activity types
        types_frame = tk.Frame(pref_frame, bg=self.theme['card_bg'])
        types_frame.pack(fill=tk.X, pady=5)
        tk.Label(types_frame, text="Preferred Activity Types:", 
                bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        
        self.type_vars = {}
        types = self.activities['Type'].unique() if 'Type' in self.activities.columns else []
        for activity_type in types:
            var = tk.IntVar()
            chk = tk.Checkbutton(types_frame, text=activity_type, variable=var,
                            bg=self.theme['card_bg'], fg=self.theme['fg'],
                            selectcolor=self.theme['bg'], activebackground=self.theme['bg'],
                            activeforeground=self.theme['fg'])
            chk.pack(anchor="w", padx=20)
            self.type_vars[activity_type] = var
        
        # Price range
        price_frame = tk.Frame(pref_frame, bg=self.theme['card_bg'])
        price_frame.pack(fill=tk.X, pady=5)
        tk.Label(price_frame, text="Price Range:", 
                bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        
        self.price_min = tk.DoubleVar(value=0)
        self.price_max = tk.DoubleVar(value=1)
        
        tk.Scale(price_frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self.price_min, label="Minimum Price",
                bg=self.theme['card_bg'], fg=self.theme['fg'],
                troughcolor=self.theme['bg'], highlightbackground=self.theme['bg'],
                activebackground=self.theme['accent']).pack(fill=tk.X)
        
        tk.Scale(price_frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self.price_max, label="Maximum Price",
                bg=self.theme['card_bg'], fg=self.theme['fg'],
                troughcolor=self.theme['bg'], highlightbackground=self.theme['bg'],
                activebackground=self.theme['accent']).pack(fill=tk.X)
        
        # Participants range (1-5)
        part_frame = tk.Frame(pref_frame, bg=self.theme['card_bg'])
        part_frame.pack(fill=tk.X, pady=5)
        tk.Label(part_frame, text="Number of Participants (1-5):",
                bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        
        self.part_min = tk.IntVar(value=1)
        self.part_max = tk.IntVar(value=5)
        
        tk.Scale(part_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                variable=self.part_min, label="Minimum",
                bg=self.theme['card_bg'], fg=self.theme['fg'],
                troughcolor=self.theme['bg'], highlightbackground=self.theme['bg'],
                activebackground=self.theme['accent']).pack(fill=tk.X)
        
        tk.Scale(part_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                variable=self.part_max, label="Maximum",
                bg=self.theme['card_bg'], fg=self.theme['fg'],
                troughcolor=self.theme['bg'], highlightbackground=self.theme['bg'],
                activebackground=self.theme['accent']).pack(fill=tk.X)
        
        # Save button
        btn_frame = tk.Frame(main_frame, bg=self.theme['bg'])
        btn_frame.pack(pady=20)
        
        save_btn = tk.Button(btn_frame, text="Save Preferences", command=self.save_preferences,
                        font=("Helvetica", 12), width=20,
                        bg=self.theme['accent'], fg='black', activebackground='#b8d8a3')
        save_btn.pack()

    def save_preferences(self):
        """Save user preferences to their account"""
        selected_types = {t for t, var in self.type_vars.items() if var.get() == 1}
        
        # Validate participants range
        part_min = min(self.part_min.get(), self.part_max.get())
        part_max = max(self.part_min.get(), self.part_max.get())
        
        self.users[self.current_user]['preferences'] = {
            'activity_types': selected_types,
            'price_range': (self.price_min.get(), self.price_max.get()),
            'participants_range': (part_min, part_max)
        }
        
        messagebox.showinfo("Success", "Preferences saved successfully!")
        self.setup_home_screen()

    def setup_home_screen(self):
        """Main application home screen"""
        self.clear_window()
        
        # Main container
        container = tk.Frame(self.root, bg=self.theme['bg'])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Navigation sidebar
        sidebar = tk.Frame(container, width=200, bg=self.theme['bg'])
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        buttons = [
            ("🏠 Home", self.setup_home_screen),
            ("🌟 Recommendations", self.setup_recommendations_screen),
            ("🎲 Random Activity", self.generate_random_activity),
            ("⚙️ Preferences", self.setup_preferences_screen),
            ("🚪 Logout", self.logout)
        ]
        
        for text, command in buttons:
            btn = tk.Button(sidebar, text=text, command=command,
                        font=("Helvetica", 12), width=20, anchor="w",
                        bg=self.theme['button_bg'], fg=self.theme['fg'],
                        activebackground='#3e3e3e', borderwidth=0)
            btn.pack(pady=5, padx=5, fill=tk.X)
        
        # Main content area with scrollbar
        content_container = tk.Frame(container, bg=self.theme['bg'])
        content_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Scrollable frame for content
        main_content = self.create_scrollable_frame(content_container)
        
        # Welcome message
        welcome_frame = tk.Frame(main_content, bg=self.theme['bg'])
        welcome_frame.pack(fill=tk.X, pady=20)
        tk.Label(welcome_frame, text=f"Welcome, {self.current_user}!",
                font=("Helvetica", 18, "bold"), bg=self.theme['bg'], fg=self.theme['accent']).pack()
        
        # User preferences
        pref_frame = tk.LabelFrame(main_content, text="Your Preferences", padx=10, pady=10,
                                bg=self.theme['card_bg'], fg=self.theme['accent'],
                                font=("Helvetica", 12, "bold"))
        pref_frame.pack(fill=tk.X, padx=20, pady=10)
        
        prefs = self.users[self.current_user]['preferences']
        tk.Label(pref_frame, text=f"Activity Types: {', '.join(prefs['activity_types']) or 'Any'}",
                bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        tk.Label(pref_frame, text=f"Price Range: ${prefs['price_range'][0]:.2f} - ${prefs['price_range'][1]:.2f}",
                bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        tk.Label(pref_frame, text=f"Participants: {prefs['participants_range'][0]} - {prefs['participants_range'][1]}",
                bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        
        # Pinned activities
        pinned_frame = tk.LabelFrame(main_content, text="📌 Pinned Activities", padx=10, pady=10,
                                bg=self.theme['card_bg'], fg=self.theme['accent'],
                                font=("Helvetica", 12, "bold"))
        pinned_frame.pack(fill=tk.X, padx=20, pady=10)
        
        pinned = self.user_pins[self.current_user]
        if pinned:
            for activity_id in pinned:
                activity = self.activities.loc[int(activity_id)]
                self.create_activity_card(pinned_frame, activity, show_buttons=False)
        else:
            tk.Label(pinned_frame, text="No pinned activities yet",
                    bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        
        # Liked activities
        liked_frame = tk.LabelFrame(main_content, text="👍 Liked Activities", padx=10, pady=10,
                                bg=self.theme['card_bg'], fg=self.theme['accent'],
                                font=("Helvetica", 12, "bold"))
        liked_frame.pack(fill=tk.X, padx=20, pady=10)
        
        liked = self.user_likes[self.current_user]
        if liked:
            for activity_id in liked:
                activity = self.activities.loc[int(activity_id)]
                self.create_activity_card(liked_frame, activity, show_buttons=False)
        else:
            tk.Label(liked_frame, text="No liked activities yet",
                    bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")

    def setup_recommendations_screen(self):
        """Screen showing personalized recommendations"""
        self.clear_window()
        
        # Main container
        container = tk.Frame(self.root, bg=self.theme['bg'])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Navigation sidebar
        sidebar = tk.Frame(container, width=200, bg=self.theme['bg'])
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        buttons = [
            ("🏠 Home", self.setup_home_screen),
            ("🌟 Recommendations", self.setup_recommendations_screen),
            ("🎲 Random Activity", self.generate_random_activity),
            ("🚪 Logout", self.logout)
        ]
        
        for text, command in buttons:
            btn = tk.Button(sidebar, text=text, command=command, 
                        font=("Helvetica", 12), width=20, anchor="w",
                        bg=self.theme['button_bg'], fg=self.theme['fg'],
                        activebackground='#3e3e3e', borderwidth=0)
            btn.pack(pady=5, padx=5, fill=tk.X)
        
        # Main content area with scrollbar
        content_container = tk.Frame(container, bg=self.theme['bg'])
        content_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Scrollable frame for content
        main_content = self.create_scrollable_frame(content_container)
        
        # Title
        title_frame = tk.Frame(main_content, bg=self.theme['bg'])
        title_frame.pack(fill=tk.X, pady=20)
        tk.Label(title_frame, text="Recommended For You", 
                font=("Helvetica", 18, "bold"), bg=self.theme['bg'], fg=self.theme['accent']).pack()
        
        # Get and display recommendations
        recommendations = self.get_recommendations(top_n=5)
        
        if not recommendations:
            tk.Label(main_content, text="No recommendations available. Try adjusting your preferences.",
                    bg=self.theme['bg'], fg=self.theme['fg']).pack(pady=20)
            return
        
        for rec in recommendations:
            self.create_activity_card(main_content, rec)

    def generate_random_activity(self):
        """Generate and display a single random activity"""
        recommendations = self.get_recommendations(top_n=1)
        if recommendations:
            self.show_activity_details(recommendations[0])
        else:
            messagebox.showerror("Error", "Couldn't generate an activity")

    def create_activity_card(self, parent, activity, show_buttons=True):
        """Create a UI card for an activity"""
        card = tk.Frame(parent, bd=0, relief=tk.GROOVE, padx=10, pady=10,
                    bg=self.theme['card_bg'])
        card.pack(fill=tk.X, padx=20, pady=10)
        
        # Header with activity name
        header = tk.Frame(card, bg=self.theme['card_bg'])
        header.pack(fill=tk.X)
        tk.Label(header, text=activity['Activity'], 
                font=("Helvetica", 14, "bold"), bg=self.theme['card_bg'], fg=self.theme['fg']).pack(side=tk.LEFT)
        
        # Activity type badge
        if 'Type' in activity:
            type_badge = tk.Label(header, text=activity['Type'], 
                                bg=self.theme['bg'], fg=self.theme['accent'], 
                                padx=5, pady=2, font=("Helvetica", 10, "bold"))
            type_badge.pack(side=tk.RIGHT)
        
        # Details row
        details = tk.Frame(card, bg=self.theme['card_bg'])
        details.pack(fill=tk.X, pady=5)
        
        if 'Price' in activity:
            price_text = f"Price: ${activity['Price']:.2f}" if activity['Price'] > 0 else "Free"
            tk.Label(details, text=price_text, bg=self.theme['card_bg'], fg=self.theme['fg']).pack(side=tk.LEFT, padx=5)
        
        if 'Participants' in activity:
            tk.Label(details, text=f"👥 {int(activity['Participants'])}/5",
                    bg=self.theme['card_bg'], fg=self.theme['fg']).pack(side=tk.LEFT, padx=5)
        
        # Description
        desc_frame = tk.Frame(card, bg=self.theme['card_bg'])
        desc_frame.pack(fill=tk.X, pady=5)
        tk.Label(desc_frame, text="Description:",
                bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        
        desc_text = ScrolledText(desc_frame, height=4, wrap=tk.WORD,
                            font=("Helvetica", 10),
                            bg=self.theme['card_bg'], fg=self.theme['fg'],
                            insertbackground='white')
        desc_text.insert(tk.END, activity['Description'])
        desc_text.config(state=tk.DISABLED)
        desc_text.pack(fill=tk.X)
        
        # Action buttons
        if show_buttons:
            btn_frame = tk.Frame(card, bg=self.theme['card_bg'])
            btn_frame.pack(anchor="e", pady=5)
            
            # Like button (green)
            like_btn = tk.Button(btn_frame, text="👍 Like",
                            command=lambda: self.like_activity(activity['id']),
                            bg=self.theme['like'], fg='white',
                            activebackground='#3e8e41')
            like_btn.pack(side=tk.LEFT, padx=5)
            
            # Dislike button (red)
            dislike_btn = tk.Button(btn_frame, text="👎 Dislike",
                                command=lambda: self.dislike_activity(activity['id']),
                                bg=self.theme['dislike'], fg='white',
                                activebackground='#c62828')
            dislike_btn.pack(side=tk.LEFT, padx=5)
            
            # Pin button (blue)
            pin_btn = tk.Button(btn_frame, text="📌 Pin", 
                              command=lambda: self.pin_activity(activity['id']),
                              bg=self.theme['pin'], fg='white',
                              activebackground='#1565c0')
            pin_btn.pack(side=tk.LEFT, padx=5)
            
            # Details button
            details_btn = tk.Button(btn_frame, text="🔍 Details",
                                  command=lambda: self.show_activity_details(activity),
                                  bg=self.theme['button_bg'], fg=self.theme['fg'],
                                  activebackground='#3e3e3e')
            details_btn.pack(side=tk.LEFT, padx=5)

    def show_activity_details(self, activity):
        """Show detailed activity view with rating options"""
        popup = tk.Toplevel(self.root)
        popup.title(activity['Activity'])
        popup.geometry("700x600")
        popup.configure(bg=self.theme['bg'])
        
        # Create the activity card
        self.create_activity_card(popup, activity, show_buttons=False)
        
        # Rating section
        rating_frame = tk.LabelFrame(popup, text="Rate This Activity", padx=10, pady=10,
                                   bg=self.theme['card_bg'], fg=self.theme['accent'],
                                   font=("Helvetica", 12, "bold"))
        rating_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(rating_frame, text="How would you rate this activity?",
                bg=self.theme['card_bg'], fg=self.theme['fg']).pack(anchor="w")
        
        rating_var = tk.IntVar(value=3)
        stars_frame = tk.Frame(rating_frame, bg=self.theme['card_bg'])
        stars_frame.pack(pady=10)
        for i in range(1, 6):
            tk.Radiobutton(stars_frame, text="★"*i, variable=rating_var, 
                          value=i, font=("Helvetica", 14),
                          bg=self.theme['card_bg'], fg=self.theme['accent'],
                          activebackground=self.theme['card_bg'],
                          selectcolor=self.theme['accent']).pack(side=tk.LEFT)
        
        # Action buttons
        btn_frame = tk.Frame(popup, bg=self.theme['bg'])
        btn_frame.pack(pady=10)
        
        submit_btn = tk.Button(btn_frame, text="Submit Rating", 
                             command=lambda: [self.rate_activity(activity['id'], rating_var.get()),
                                            popup.destroy()],
                             bg=self.theme['accent'], fg='black',
                             activebackground='#b8d8a3')
        submit_btn.pack(side=tk.LEFT, padx=10)
        
        close_btn = tk.Button(btn_frame, text="Close", command=popup.destroy,
                            bg=self.theme['button_bg'], fg=self.theme['fg'],
                            activebackground='#3e3e3e')
        close_btn.pack(side=tk.LEFT, padx=10)

    def clear_window(self):
        """Clear all widgets from the root window"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def logout(self):
        """Handle user logout"""
        self.current_user = None
        self.setup_login_screen()

if __name__ == "__main__":
    app = ActivityRecommender()