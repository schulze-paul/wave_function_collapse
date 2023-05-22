import time
import numpy as np
import matplotlib.pyplot as plt

class Tile:

    def __init__(self, tile_data:np.array):
        self.tile_data = tile_data

    def rotate(self, n_rotations=1):
        for _ in range(n_rotations):
            self.tile_data = np.rot90(self.tile_data)
    
    def flip(self):
        self.tile_data = np.fliplr(self.tile_data)

    def __eq__(self, other):
        return np.array_equal(self.tile_data, other.tile_data)
    
    def __hash__(self):
        return hash(self.tile_data.tobytes())
    
    def __str__(self):
        return str(self.tile_data)
        tile_string = ""
        for i in range(0, 3):
            for j in range(0, 3):
                tile_string += "  " if self.tile_data[i][j] == 0 else "██"
            tile_string += "\n"
        return tile_string
    
    def __repr__(self):
        return self.__str__()
    
    def __getitem__(self, key):
        return self.tile_data[key]
    
    def get_connection(self, direction):
        """get the connection on a given side."""
        connector_middle_pos = direction + np.array([1, 1])
        return self.tile_data[connector_middle_pos[0], connector_middle_pos[1]]
        
    def get_possible_neighbors(self, direction, all_tiles):
        """"""
        possible_neighbors = []

        connection = self.get_connection(direction)
        for other_tile in all_tiles:
            connection_other_tile = other_tile.get_connection(-direction)
            if np.allclose(connection, connection_other_tile):
                possible_neighbors.append(other_tile)
        return possible_neighbors
    
    def copy(self):
        return Tile(self.tile_data.copy())


class TileGrid:
    pass

class Cell:
    """A cell in the grid."""

    def __init__(self, grid: TileGrid, position: tuple):
        self.grid = grid
        self.position = position
        self.tile = None
        self.collapsed = False
        self.possible_tiles = grid.all_tiles.copy()

    def get_neighbor(self, direction):
        """Get the neighbor in a given direction."""
        neighbor_position = tuple(np.array(self.position) + direction)
        if self.grid.position_in_grid(neighbor_position):
            return self.grid.get_cell(*neighbor_position)

    def __len__(self):
        return len(self.possible_tiles)
    
    def __getitem__(self, key):
        return self.possible_tiles[key]

    def set_possible_neighbors(self, allowed_tiles: list = None, in_direction: tuple = None):
        """Set the possible neighbors for this cell."""
        # if the cell is collapsed, only the current tile is allowed
        if self.collapsed or allowed_tiles is None:
            allowed_tiles = [self.tile]

        if len(allowed_tiles) == 1 and not self.collapsed:
            # only one tile is allowed: collapse the cell
            self.set_tile(allowed_tiles[0])
        
        if len(set(allowed_tiles)) == len(self.grid.all_tiles):
            # no restrictions on this cell, so no need to inform neighbors
            return

        # incorporate the restrictions from the allowed tiles
        if not self.collapsed:
            intersection = []
            for tile in self.possible_tiles:
                if tile in allowed_tiles:
                    intersection.append(tile)
            if len(intersection) == len(self.possible_tiles):
                return # no new restrictions on this cell, so no need to inform neighbors
            self.possible_tiles = intersection

        for direction in DIRECTIONS:
            if in_direction is not None and np.allclose(direction, -in_direction):
                continue # don't inform the cell we got the restrictions from
            neighbor_cell = self.get_neighbor(direction)
            if neighbor_cell is None or neighbor_cell.collapsed:
                continue # no neighbor in this direction, or neighbor is collapsed
            
            # do the tiles allowed in this cell have restrictions on the possible neighbors?
            tile_restrictions = []
            
            for tile in self.possible_tiles:
                # any tile that is allowed by the cells allowed in this cell is allowed 
                for tile_restriction in tile.get_possible_neighbors(direction, self.grid.all_tiles):
                    tile_restrictions.append(tile_restriction)
            tile_restrictions = list(set(tile_restrictions)) # remove duplicates

            # if any tile is allowed, no need to inform neighbor
            if len(tile_restrictions) == len(self.grid.all_tiles):
                continue 
            
            # inform the neighbor of the restrictions
            neighbor_cell.set_possible_neighbors(tile_restrictions, direction)

    def set_tile(self, tile=None):
        """Set the tile for this cell."""
        if tile is None:
            tile = np.random.choice(self.possible_tiles)
        self.tile = tile
        self.collapsed = True
        self.possible_tiles = [tile]
        self.grid.collapsed_cells.insert(0, self)

class TileGrid:

    def __init__(self, n_rows: int, n_cols: int, all_tiles: list) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.all_tiles = all_tiles
        self.cells = [[Cell(self, (row, col)) for col in range(n_cols)] for row in range(n_rows)]
        self.collapsed_cells = []

    def get_cell(self, row, col):
        return self.cells[row][col]
    
    def step(self):
        entropy = np.array([[len(self.cells[row][col].possible_tiles) for col in range(self.n_cols)] for row in range(self.n_rows)])
        if max(entropy.flatten()) == 1:
            return False # no more steps possible
        min_entropy = np.min(entropy[entropy>1])
        min_entropy_indices = np.argwhere(entropy == min_entropy)

        # select a random cell with minimum entropy
        row, col = min_entropy_indices[np.random.randint(len(min_entropy_indices))]
        self.cells[row][col].set_tile()
        
        if self.all_cells_collapsed():
            return False

        for cell in self.collapsed_cells:
            cell.set_possible_neighbors()
        return True

    def all_cells_collapsed(self):
        return len(self.collapsed_cells) == self.n_rows * self.n_cols

    def position_in_grid(self, position):
        return position[0] >= 0 and position[0] < self.n_rows and position[1] >= 0 and position[1] < self.n_cols
    
    def get_neighbors(self, row, col):
        neighbors = []
        for direction in DIRECTIONS:
            neighbor_position = (row + direction[0], col + direction[1])
            if self.position_in_grid(neighbor_position):
                neighbors.append(self.cells[neighbor_position[0]][neighbor_position[1]])
            else:
                neighbors.append(None)

        return neighbors
        
    

    def __str__(self):
        grid_string = "="*(self.n_cols*3*2) + "\n"
        # go through the rows
        for row in range(self.n_rows):
            # add the top border
            for tile_row in range(3):
                for col in range(self.n_cols):
                    if len(self.cells[row][col]) > 1:
                        grid_string += "XXXXXX"
                    else:
                        for tile_col in range(3):
                            try:
                                grid_string += "  " if self.cells[row][col][0][tile_row][tile_col] == 0 else "██"
                            except IndexError:
                                print(row, col, len(self.cells[row][col]))
                                raise IndexError
                grid_string += "\n"

        return grid_string[:-2] # remove last newline

    def show(self):
        # show the grid as image
        grid_image = 0.5*np.ones((self.n_rows*3, self.n_cols*3))
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if len(self.cells[row][col]) > 1:
                    continue
                for tile_row in range(3):
                    for tile_col in range(3):
                        grid_image[row*3+tile_row, col*3+tile_col] = self.cells[row][col][0][tile_row][tile_col]

        plt.imshow(grid_image, cmap="gray")
        plt.show()
    
    def show_options(self):
        # show the grid as image where each cell shows the possible options
        grid_image = 0.5*np.ones((self.n_rows*9, self.n_cols*9))
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                possibilities_image = 0.5*np.ones((9, 9))
                if len(self.cells[row][col]) == len(self.all_tiles):
                    continue
                if len(self.cells[row][col]) == 1:
                    possibilities_image = self.cells[row][col][0].tile_data.repeat(3, axis=0).repeat(3, axis=1)
                else:
                    for index, tile in enumerate(self.cells[row][col]):
                        x_offset = (index % 3) * 3
                        y_offset = (index // 3) * 3
                        possibilities_image[y_offset:y_offset+3, x_offset:x_offset+3] = tile.tile_data
                grid_image[row*9:(row+1)*9, col*9:(col+1)*9] = possibilities_image

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(grid_image, cmap="gray", vmin=0, vmax=1)
        plt.show()

def get_similar_tiles(tile_data):
    tiles = []
    for r in range(0, 4):
        tile = Tile(tile_data)
        tile.rotate(r)
        tiles.append(tile)
    return list(set(tiles))


def main():
    # north, east, south, west
    DIRECTIONS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

    base_tiles = {
        " ": np.array([
            [0, 0, 0,],
            [0, 0, 0,],
            [0, 0, 0,],
        ]),
        "+": np.array([
            [0, 1, 0,],
            [1, 1, 1,],
            [0, 1, 0,],
        ]),
        "T": np.array([
            [0, 0, 0],
            [1, 1, 1,],
            [0, 1, 0],
        ]),
        "L": np.array([
            [0, 0, 0],
            [1, 1, 0,],
            [0, 1, 0],
        ]),
        "-": np.array([
            [0, 0, 0],
            [1, 1, 1,],
            [0, 0, 0],
        ]),
    }

    tile_for_this_run = [" ", "L", "+", "-"]

    all_tiles = []
    for tile_name in tile_for_this_run:    
        all_tiles += get_similar_tiles(base_tiles[tile_name])

    grid = TileGrid(40, 40, all_tiles)
    while grid.step():
        pass
    grid.show_options()

    if __name__ == "__main__":
        main()
